import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_clustering


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class SharedSemanticProjector(nn.Module):
    """One residual semantic filter shared by every view.

    View encoders remain responsible for reconstruction.  This projector is
    shared across views so the clustering space cannot acquire a separate
    dataset- or view-specific prediction branch.  Its final layer is zero
    initialised, making the enabled model exactly reproduce the pretrained
    latent representation before contrastive fine-tuning starts.
    """

    def __init__(self, feature_dim, hidden_dim=None):
        super(SharedSemanticProjector, self).__init__()
        hidden_dim = int(hidden_dim or feature_dim)
        self.norm = nn.LayerNorm(int(feature_dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(feature_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(feature_dim)),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z, scale=1.0):
        return z + float(scale) * self.mlp(self.norm(z))


class ZeroInitialisedResidualBranch(nn.Module):
    """A residual latent branch whose enabled starting point is identity/zero."""

    def __init__(self, feature_dim, hidden_dim=None):
        super(ZeroInitialisedResidualBranch, self).__init__()
        hidden_dim = int(hidden_dim or feature_dim)
        self.norm = nn.LayerNorm(int(feature_dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(feature_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(feature_dim)),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z):
        return self.mlp(self.norm(z))


class IdentityInitialisedExpert(nn.Module):
    """A nonlinear expert whose initial mapping is exactly the identity.

    The final projection is zero initialised so a cached pretrained autoencoder can
    be upgraded after pretraining without an artificial representation jump.
    Different hidden-layer initialisations break expert symmetry after the
    first gradient update while preserving that exact starting endpoint.
    """

    def __init__(self, feature_dim, hidden_dim=None):
        super(IdentityInitialisedExpert, self).__init__()
        feature_dim = int(feature_dim)
        hidden_dim = int(hidden_dim or feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features, residual_scale=1.0):
        return features + float(residual_scale) * self.mlp(
            self.norm(features)
        )


class ContextAwareSharedPrivateMoE(nn.Module):
    """Latent D-MoE with cross-view context-aware shared routing.

    A public expert pool is reused by every view.  Its router sees both the
    current latent code and a shared, per-sample context aggregated from all
    views.  Each view also owns a private residual-expert pool whose router
    only sees that view.  Shared experts start as identity mappings while
    private experts start at zero; reconstruction consumes ``shared +
    private``.  Thus enabling this module after autoencoder pretraining
    initially reproduces the pretrained representation and decoder input
    exactly, without duplicating the same latent in both branches.
    """

    def __init__(
        self,
        num_views,
        feature_dim,
        hidden_dim=None,
        shared_experts=4,
        private_experts=2,
        top_k=2,
    ):
        super(ContextAwareSharedPrivateMoE, self).__init__()
        self.num_views = int(num_views)
        self.feature_dim = int(feature_dim)
        self.shared_expert_count = int(shared_experts)
        self.private_expert_count = int(private_experts)
        self.top_k = int(top_k)
        if self.num_views < 2:
            raise ValueError("context-aware MoE requires at least two views")
        if self.shared_expert_count < 1 or self.private_expert_count < 1:
            raise ValueError("expert counts must be positive")
        if self.top_k < 1 or self.top_k > self.shared_expert_count:
            raise ValueError("top_k must lie within the shared expert pool")

        hidden_dim = int(hidden_dim or feature_dim)
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
        )
        self.shared_experts = nn.ModuleList(
            IdentityInitialisedExpert(
                self.feature_dim, hidden_dim=hidden_dim
            )
            for _ in range(self.shared_expert_count)
        )
        self.shared_router = nn.Sequential(
            nn.LayerNorm(2 * self.feature_dim),
            nn.Linear(2 * self.feature_dim, self.shared_expert_count),
        )
        self.private_experts = nn.ModuleList(
            nn.ModuleList(
                ZeroInitialisedResidualBranch(
                    self.feature_dim, hidden_dim=hidden_dim
                )
                for _ in range(self.private_expert_count)
            )
            for _ in range(self.num_views)
        )
        self.private_routers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.private_expert_count),
            )
            for _ in range(self.num_views)
        )

    @staticmethod
    def _topk_softmax(logits, top_k):
        probability = F.softmax(logits, dim=1)
        if int(top_k) >= int(probability.size(1)):
            return probability
        values, indices = torch.topk(
            probability, int(top_k), dim=1, largest=True, sorted=False
        )
        sparse = torch.zeros_like(probability).scatter(1, indices, values)
        return sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _mixture(features, experts, routing, residual_scale):
        # Mix only residuals and add the identity once.  Besides avoiding a
        # redundant copy of the base code per expert, this makes the
        # zero-initialised endpoint bit-for-bit equal to ``features`` even
        # when softmax weights sum to one only up to floating-point error.
        expert_residuals = torch.stack(
            [
                expert(features, residual_scale=residual_scale) - features
                for expert in experts
            ],
            dim=1,
        )
        return features + torch.sum(
            expert_residuals * routing.unsqueeze(2), dim=1
        )

    @staticmethod
    def _residual_mixture(features, experts, routing, residual_scale):
        """Mix zero-initialised private residuals without copying features."""
        expert_residuals = torch.stack(
            [expert(features) for expert in experts], dim=1
        )
        return float(residual_scale) * torch.sum(
            expert_residuals * routing.unsqueeze(2), dim=1
        )

    def forward(self, encoded_views, residual_scale=1.0):
        if len(encoded_views) != self.num_views:
            raise ValueError("encoded view count does not match the MoE")
        first_shape = tuple(encoded_views[0].shape)
        if len(first_shape) != 2 or first_shape[1] != self.feature_dim:
            raise ValueError("encoded views must have shape [B, feature_dim]")
        if any(tuple(features.shape) != first_shape for features in encoded_views):
            raise ValueError("all encoded views must share shape [B,D]")

        context_blocks = [
            self.context_encoder(features) for features in encoded_views
        ]
        global_context = torch.stack(context_blocks, dim=0).mean(dim=0)
        shared_views = []
        private_views = []
        shared_routing = []
        private_routing = []
        private_top_k = min(self.top_k, self.private_expert_count)
        for view, features in enumerate(encoded_views):
            shared_gate = self._topk_softmax(
                self.shared_router(
                    torch.cat([features, global_context], dim=1)
                ),
                self.top_k,
            )
            private_gate = self._topk_softmax(
                self.private_routers[view](features), private_top_k
            )
            shared_views.append(
                self._mixture(
                    features,
                    self.shared_experts,
                    shared_gate,
                    residual_scale,
                )
            )
            private_views.append(
                self._residual_mixture(
                    features,
                    self.private_experts[view],
                    private_gate,
                    residual_scale,
                )
            )
            shared_routing.append(shared_gate)
            private_routing.append(private_gate)
        return (
            shared_views,
            private_views,
            shared_routing,
            private_routing,
        )


class Network(nn.Module):
    def __init__(
        self,
        num_views,
        num_samples,
        num_clusters,
        input_size,
        feature_dim,
        memory_positive_k=9,
        memory_negative_k=64,
        normalize_fusion_features=False,
        enable_semantic_projector=False,
        semantic_projector_hidden_dim=None,
        semantic_projector_scale=None,
        shared_private_scale=0.0,
        enable_shared_private=False,
        shared_private_hidden_dim=None,
        context_aware_moe=False,
        moe_shared_experts=4,
        moe_private_experts=2,
        moe_top_k=2,
        enable_canonical_correlation_head=False,
        canonical_correlation_dim=32,
        enable_multilevel_heads=False,
        multilevel_projection_dim=128,
    ):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for i in range(num_views):
            self.encoders.append(Encoder(input_size[i], feature_dim))
            self.decoders.append(Decoder(input_size[i], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.num_views = num_views
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.feature_dim = int(feature_dim)
        # Keep reconstruction pretraining and the released endpoint intact.
        # When enabled after pretraining, every view contributes a direction
        # to the convex consensus; arbitrary encoder norms can no longer
        # silently multiply the declared reliability weights.
        self.normalize_fusion_features = bool(normalize_fusion_features)
        self.semantic_projector_scale = float(
            1.0
            if semantic_projector_scale is None
            and bool(enable_semantic_projector)
            else (semantic_projector_scale or 0.0)
        )
        self.enable_semantic_projector = (
            self.semantic_projector_scale > 0.0
        )
        # The shared projector is always present.  A zero residual scale is
        # the exact identity endpoint of the same model equation rather than
        # a dataset-specific architectural branch.
        self.semantic_projector = SharedSemanticProjector(
            feature_dim, hidden_dim=semantic_projector_hidden_dim
        )
        self.shared_private_scale = float(shared_private_scale)
        self.context_aware_moe = bool(context_aware_moe)
        if self.context_aware_moe and self.shared_private_scale <= 0.0:
            raise ValueError(
                "context-aware MoE requires positive shared_private_scale"
            )
        self.enable_shared_private = bool(enable_shared_private) or (
            self.shared_private_scale > 0.0
        )
        if self.context_aware_moe:
            self.context_moe = ContextAwareSharedPrivateMoE(
                self.num_views,
                feature_dim,
                hidden_dim=shared_private_hidden_dim,
                shared_experts=moe_shared_experts,
                private_experts=moe_private_experts,
                top_k=moe_top_k,
            )
            self.shared_residual = None
            self.private_residuals = nn.ModuleList()
        elif self.enable_shared_private:
            self.context_moe = None
            self.shared_residual = ZeroInitialisedResidualBranch(
                feature_dim, hidden_dim=shared_private_hidden_dim
            )
            self.private_residuals = nn.ModuleList(
                ZeroInitialisedResidualBranch(
                    feature_dim, hidden_dim=shared_private_hidden_dim
                )
                for _ in range(self.num_views)
            )
        else:
            # Avoid even consuming RNG for a fully disabled endpoint.  The
            # adaptive endpoint explicitly opts in to these zero-initialised
            # branches while starting with a zero runtime scale.
            self.context_moe = None
            self.shared_residual = None
            self.private_residuals = nn.ModuleList()
        self.enable_multilevel_heads = bool(enable_multilevel_heads)
        self.multilevel_projection_dim = int(multilevel_projection_dim)
        self.enable_canonical_correlation_head = bool(
            enable_canonical_correlation_head
        )
        self.canonical_correlation_dim = min(
            int(canonical_correlation_dim),
            self.feature_dim,
            self.num_clusters,
        )
        if (
            self.enable_canonical_correlation_head
            and self.canonical_correlation_dim < 1
        ):
            raise ValueError("canonical correlation dimension must be positive")
        # Like the other auxiliary heads, these are created only after the
        # cached reconstruction stage.  The disabled endpoint therefore has
        # exactly the original parameters and RNG trajectory.
        self.canonical_correlation_heads = None
        # Heads are initialised only after reconstruction pretraining.  This
        # makes cache reuse and a from-scratch run identical and prevents an
        # auxiliary stage from perturbing the released pretraining RNG stream.
        self.multilevel_instance_head = None
        self.multilevel_cluster_head = None
        # Ephemeral outputs used only by the decoupling regularizer.  They are
        # deliberately absent from checkpoints and inference APIs.
        self.last_private_features = []
        self.last_shared_routing = []
        self.last_private_routing = []

        self.register_buffer(
            "psedo_labels", torch.zeros(self.num_samples, dtype=torch.long)
        )
        self.register_buffer(
            "trust_scores", torch.ones(self.num_samples, dtype=torch.float32)
        )
        self.register_buffer(
            "margin_scores", torch.zeros(self.num_samples, dtype=torch.float32)
        )
        self.register_buffer(
            "consistency_scores", torch.zeros(self.num_samples, dtype=torch.float32)
        )
        self.register_buffer(
            "soft_targets",
            torch.zeros(
                self.num_samples, self.num_clusters, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "barycentric_targets",
            torch.full(
                (self.num_samples, self.num_clusters),
                1.0 / float(self.num_clusters),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "prototype_centers",
            torch.zeros(self.num_clusters, feature_dim, dtype=torch.float32),
        )
        # Training-only counter for the aligned temporal pseudo teacher.  The
        # teacher targets and prototypes reuse the public buffers above; this
        # counter is deliberately non-persistent because it is irrelevant to
        # inference and keeps old pretraining checkpoints compatible.
        self.register_buffer(
            "pseudo_teacher_updates",
            torch.tensor(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "sample_reliability",
            torch.full(
                (self.num_samples, self.num_views),
                1.0 / float(self.num_views),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "global_view_gate",
            torch.full(
                (self.num_views,),
                1.0 / float(self.num_views),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "global_view_quality",
            torch.full(
                (self.num_views,),
                1.0 / float(self.num_views),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "reliability_power",
            torch.tensor(1.0, dtype=torch.float32),
        )
        self.register_buffer(
            "reliability_updates",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "anchor_view_index",
            torch.tensor(-1, dtype=torch.long),
        )
        self.register_buffer(
            "anchor_expert_mass",
            torch.tensor(0.0, dtype=torch.float32),
        )
        # Training-only sparse global topology memory.  These buffers are not
        # persisted because final inference uses only the learned encoders and
        # reliability weights; excluding them keeps large-dataset checkpoints
        # compact while still moving the teacher state with ``model.to``.
        self.register_buffer(
            "global_feature_bank",
            torch.zeros(self.num_samples, feature_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "global_view_feature_banks",
            torch.zeros(
                self.num_views,
                self.num_samples,
                feature_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "raw_topology_bank",
            torch.zeros(self.num_samples, feature_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "raw_topology_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "raw_view_relation_bank",
            torch.zeros(
                self.num_views,
                self.num_samples,
                feature_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "raw_view_relation_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "raw_view_candidate_indices",
            torch.zeros(
                self.num_views,
                self.num_samples,
                int(memory_positive_k),
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "raw_view_candidate_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "memory_positive_indices",
            torch.zeros(
                self.num_samples, int(memory_positive_k), dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "memory_positive_weights",
            torch.zeros(
                self.num_samples, int(memory_positive_k), dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "fixed_memory_positive_indices",
            torch.zeros(
                self.num_samples, int(memory_positive_k), dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "fixed_memory_positive_weights",
            torch.zeros(self.num_samples, int(memory_positive_k)),
            persistent=False,
        )
        self.register_buffer(
            "fixed_memory_positive_enabled",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "memory_negative_indices",
            torch.zeros(
                self.num_samples, int(memory_negative_k), dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "memory_negative_weights",
            torch.zeros(
                self.num_samples, int(memory_negative_k), dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "memory_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "memory_last_refresh",
            torch.tensor(-1, dtype=torch.long),
            persistent=False,
        )

        self.weights = nn.Parameter(torch.full((self.num_views,), 1 / self.num_views), requires_grad=True)

    def forward(self, xs):
        if self.context_moe is not None:
            encoded_views = [
                self.encoders[index](xs[index])
                for index in range(self.num_views)
            ]
            (
                shared_views,
                private_views,
                shared_routing,
                private_routing,
            ) = self.context_moe(
                encoded_views, residual_scale=self.shared_private_scale
            )
            # The shared stream is identity-initialised and the private stream
            # is zero-initialised, so their sum exactly recovers the cached
            # autoencoder latent at the enabled starting point.
            reconstruction_latents = [
                shared + private
                for shared, private in zip(shared_views, private_views)
            ]
            xrs = [
                self.decoders[index](reconstruction_latents[index])
                for index in range(self.num_views)
            ]
            zs = [
                self.semantic_projector(
                    shared, scale=self.semantic_projector_scale
                )
                for shared in shared_views
            ]
            self.last_private_features = private_views
            self.last_shared_routing = shared_routing
            self.last_private_routing = private_routing
            return xrs, zs

        xrs = []
        zs = []
        private_features = []
        for i in range(self.num_views):
            encoded = self.encoders[i](xs[i])
            if self.shared_private_scale > 0.0:
                shared_delta = (
                    self.shared_private_scale * self.shared_residual(encoded)
                )
                private = (
                    self.shared_private_scale
                    * self.private_residuals[i](encoded)
                )
                # Clustering consumes only the shared branch.  Reconstruction
                # consumes shared plus private information, so alignment can
                # improve semantic invariance without forcing the shared code
                # alone to retain every view-specific reconstruction detail.
                z = encoded + shared_delta
                reconstruction_latent = z + private
            else:
                # This branch is intentionally identical to the released
                # network, including operation order, for a strict endpoint.
                z = encoded
                private = torch.zeros_like(encoded)
                reconstruction_latent = encoded
            xr = self.decoders[i](reconstruction_latent)
            z = self.semantic_projector(
                z, scale=self.semantic_projector_scale
            )
            xrs.append(xr)
            zs.append(z)
            private_features.append(private)
        self.last_private_features = private_features
        self.last_shared_routing = []
        self.last_private_routing = []
        return xrs, zs

    def get_weights(self):
        softmax_weights = torch.softmax(self.weights, dim=0)
        weights = softmax_weights / torch.sum(softmax_weights)
        return weights

    def multilevel_outputs(self, view_features):
        """Return decoupled instance projections and cluster logits."""
        if (
            not self.enable_multilevel_heads
            or self.multilevel_instance_head is None
            or self.multilevel_cluster_head is None
        ):
            raise RuntimeError("multilevel heads are not initialised")
        projections = [
            F.normalize(self.multilevel_instance_head(z), dim=1)
            for z in view_features
        ]
        logits = [self.multilevel_cluster_head(z) for z in view_features]
        return projections, logits

    def canonical_correlation_outputs(self, view_features):
        """Project each view into its own low-rank CCA coordinate system."""
        if (
            not self.enable_canonical_correlation_head
            or self.canonical_correlation_heads is None
        ):
            raise RuntimeError("canonical correlation heads are not initialised")
        if len(view_features) != self.num_views:
            raise ValueError("view feature count does not match the model")
        return [
            self.canonical_correlation_heads[index](features)
            for index, features in enumerate(view_features)
        ]

    def initialize_canonical_correlation_heads(self, seed=0):
        """Create view-specific CCA heads without advancing experiment RNG."""
        if not self.enable_canonical_correlation_head:
            return []
        if self.canonical_correlation_heads is not None:
            return list(self.canonical_correlation_heads.parameters())
        device = self.weights.device
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            self.canonical_correlation_heads = nn.ModuleList(
                nn.Linear(
                    self.feature_dim,
                    self.canonical_correlation_dim,
                    bias=False,
                ).to(device)
                for _ in range(self.num_views)
            )
            for head in self.canonical_correlation_heads:
                nn.init.orthogonal_(head.weight)
        return list(self.canonical_correlation_heads.parameters())

    def initialize_multilevel_heads(self, seed=0):
        """Create auxiliary heads without advancing the experiment RNG."""
        if not self.enable_multilevel_heads:
            return []
        if self.multilevel_instance_head is not None:
            return list(self.multilevel_instance_head.parameters()) + list(
                self.multilevel_cluster_head.parameters()
            )
        device = self.weights.device
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            self.multilevel_instance_head = nn.Linear(
                self.feature_dim, self.multilevel_projection_dim
            ).to(device)
            self.multilevel_cluster_head = nn.Linear(
                self.feature_dim, self.num_clusters
            ).to(device)
        return list(self.multilevel_instance_head.parameters()) + list(
            self.multilevel_cluster_head.parameters()
        )

    def effective_reliability(self, reliability, power=None):
        """Sharpen reliability without changing its per-sample normalization."""
        reliability = reliability.to(dtype=self.weights.dtype, device=self.weights.device)
        exponent = self.reliability_power if power is None else float(power)
        reliability = reliability.clamp_min(1e-6).pow(exponent)
        return reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def robust_view_quality(reliability, trim_fraction=0.10):
        """Aggregate sample-view reliability without tail domination."""
        values = reliability.detach().float()
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("reliability must be a non-empty [N, V] tensor")
        trim = int(values.shape[0] * float(trim_fraction))
        if trim > 0 and 2 * trim < values.shape[0]:
            values = torch.sort(values, dim=0).values[trim:-trim]
        quality = values.mean(dim=0).clamp_min(1e-8)
        return quality / quality.sum().clamp_min(1e-8)

    @torch.no_grad()
    def update_global_view_gate(self, reliability, temperature=1e6):
        """Map robust dataset-level reliability to one global view gate."""
        temperature = float(temperature)
        if (
            not torch.isfinite(torch.tensor(temperature))
            or temperature <= 0.0
        ):
            raise ValueError("global view-gate temperature must be positive")
        view_quality = self.robust_view_quality(reliability)
        # ``1e6`` is the declared uniform-training endpoint.  Materialising
        # that limit exactly prevents sub-ULP view biases from being amplified
        # over a long pseudo-label training trajectory.
        if temperature >= 1e5:
            gate = torch.full_like(
                view_quality, 1.0 / float(view_quality.numel())
            )
        else:
            gate = torch.softmax(view_quality.log() / temperature, dim=0)
        self.global_view_quality.copy_(
            view_quality.to(
                device=self.global_view_quality.device,
                dtype=self.global_view_quality.dtype,
            )
        )
        self.global_view_gate.copy_(
            gate.to(
                device=self.global_view_gate.device,
                dtype=self.global_view_gate.dtype,
            )
        )
        return self.global_view_gate

    def fusion(self, zs, reliability=None, reliability_power=None):
        fusion_zs = (
            [F.normalize(z, p=2, dim=1) for z in zs]
            if self.normalize_fusion_features
            else zs
        )
        weights = self.get_weights()
        global_gate = self.global_view_gate.to(
            dtype=weights.dtype, device=weights.device
        )
        if reliability is None:
            effective = weights * global_gate
            effective = effective / effective.sum().clamp_min(1e-8)
            effective = effective.view(1, -1).expand(zs[0].size(0), -1)
        else:
            reliability = self.effective_reliability(
                reliability, power=reliability_power
            ).to(zs[0])
            effective = (
                reliability
                * weights.view(1, -1)
                * global_gate.view(1, -1)
            )
            effective = effective / effective.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weighted_zs = [
            z * effective[:, index : index + 1]
            for index, z in enumerate(fusion_zs)
        ]
        stacked_zs = torch.stack(weighted_zs)
        common_z = torch.sum(stacked_zs, dim=0)
        return common_z

    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(self.num_clusters, n_samples, device=x.device, dtype=x.dtype)
            weight[psedo_labels, torch.arange(n_samples, device=x.device)] = 1
        weight = F.normalize(weight, p=1, dim=1)
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers

    def clustering(self, features):
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': self.num_clusters,
            'verbose': False
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features.to(dtype=torch.float64))
        return psedo_labels
