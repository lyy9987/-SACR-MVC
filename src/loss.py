import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(
        self,
        batch_size,
        num_clusters,
        temperature_l,
        temperature_f,
        cluster_occupancy_weighting=False,
    ):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.temperature_l = temperature_l
        self.temperature_f = temperature_f
        self.cluster_occupancy_weighting = bool(
            cluster_occupancy_weighting
        )
        self.similarity = nn.CosineSimilarity(dim=2)

    @staticmethod
    def reliability_weighted_logits(similarity, weight, mask):
        """Apply reliability as multiplicative mass in exponential space."""
        log_weight = torch.log(weight.clamp_min(1e-8))
        masked_value = torch.full_like(similarity, float(torch.log(torch.tensor(1e-45))))
        return torch.where(mask, similarity + log_weight, masked_value)

    def feature_loss(
        self,
        zi,
        z,
        w,
        y_pse,
        trust_scores=None,
        trust_pair_lambda=0.5,
        reliability_graph=None,
        graph_lambda=0.0,
        normalized_multi_positive=False,
    ):
        """
        具有样本可信度控制的特征对比损失。
        - 高可信样本对：保留较强正对齐；
        - 低可信样本对：降低伪标签错误带来的错误正对齐/错误负对排斥；
        - 注意：这个损失仍然服务于全体样本，不再只优化前 10/20/30%。
        """
        N = z.size(0)
        device = z.device

        # ``w`` may be either the original binary KNN graph or a continuously
        # routed graph.  Treat it as edge confidence rather than discarding the
        # reliability magnitude with ``bool()``.
        neighbour_strength = w.to(device).float().clamp(0.0, 1.0)
        w_bool = neighbour_strength > 0.0
        eye = torch.eye(N, dtype=torch.bool, device=device)
        binary_graph = bool(
            torch.all((neighbour_strength == 0.0) | (neighbour_strength == 1.0)).item()
        )
        pos_mask = (w_bool | eye) & y_pse.bool()
        if binary_graph:
            # Preserve the ordinary KNN objective exactly for binary
            # and hard-routed graphs.  This makes the soft-graph ablation a
            # controlled change rather than silently altering every baseline.
            neg_mask = (~(w_bool | eye)) & (~y_pse.bool())
            positive_graph_mass = torch.ones_like(neighbour_strength)
            negative_graph_mass = torch.ones_like(neighbour_strength)
        else:
            neg_mask = (~eye) & (~y_pse.bool())
            positive_graph_mass = (neighbour_strength + eye.float()).clamp(max=1.0)
            negative_graph_mass = (1.0 - neighbour_strength) * (~eye).float()

        cross_view_distance = self.similarity(zi.unsqueeze(1), z.unsqueeze(0)) / self.temperature_f
        inter_view_distance = self.similarity(zi.unsqueeze(1), zi.unsqueeze(0)) / self.temperature_f

        if trust_scores is None:
            pos_weight = pos_mask.float() * positive_graph_mass
            neg_weight = neg_mask.float() * negative_graph_mass
            pair_uncertain = None
        else:
            trust_scores = trust_scores.to(device).float().clamp(0.0, 1.0)
            pair_trust = torch.sqrt(trust_scores.unsqueeze(1) * trust_scores.unsqueeze(0))
            pair_uncertain = torch.sqrt((1.0 - trust_scores).unsqueeze(1) * (1.0 - trust_scores).unsqueeze(0))

            # 正样本：低可信伪标签不强行拉近，避免把边界样本拉到错误簇。
            pos_weight = (
                pos_mask.float()
                * positive_graph_mass
                * (0.35 + 0.65 * pair_trust)
            )
            # 负样本：低可信样本不强行推远，避免 false negative。
            neg_weight = (
                neg_mask.float()
                * negative_graph_mass
                * (1.0 - trust_pair_lambda * pair_uncertain)
            )
            neg_weight = torch.clamp(neg_weight, min=0.10, max=1.0)

        # BGCR-v38：高阶可靠图修正。R_ij 越高，越不应该把 i,j 当作强负样本；
        # 若伪标签把可靠邻居分到不同簇，则降低负样本排斥；若伪标签相同，则增强正样本吸引。
        # 该项借鉴 DIVIDE 的“用高阶随机游走识别局部邻域里的假负样本/假正样本”思想，
        # 但实现为轻量图权重，不直接替换当前特征对比主损失。
        if reliability_graph is not None and float(graph_lambda) > 0:
            R = reliability_graph.to(device).float().clamp(0.0, 1.0)
            R = R * (~eye).float()
            if trust_scores is not None:
                R = R * (0.35 + 0.65 * torch.sqrt(trust_scores.unsqueeze(1) * trust_scores.unsqueeze(0)))
            # 增强可靠正关系；抑制可靠样本被当成负关系推出去。
            pos_weight = torch.maximum(pos_weight, float(graph_lambda) * R * (w_bool | eye).float())
            neg_weight = neg_weight * (1.0 - torch.clamp(float(graph_lambda) * R, 0.0, 0.90))

        # A reliability coefficient is a multiplicative mass in exp-space:
        #   log(w * exp(sim / tau)) = log(w) + sim / tau.
        # Multiplying the logit itself by w is not equivalent and even makes a
        # negative logit larger when w < 1.  That inherited implementation was
        # the main mathematical bug identified in the audit.
        negative_cross = self.reliability_weighted_logits(
            cross_view_distance, neg_weight, neg_mask
        )
        negative_inter = self.reliability_weighted_logits(
            inter_view_distance, neg_weight, neg_mask
        )

        if bool(normalized_multi_positive):
            # Multi-positive InfoNCE/SupCon form.  Every anchor contributes
            # once, independent of its KNN degree or pseudo-cluster size.  The
            # cross-view positives and both negative banks share one partition
            # function, rather than adding an unnormalised attraction term to
            # a separate negative-only log-sum-exp.
            positive_mass = pos_weight.sum(dim=1)
            valid_anchor = positive_mass > 0.0
            positive_mean = (
                pos_weight * cross_view_distance
            ).sum(dim=1) / positive_mass.clamp_min(1e-8)
            positive_cross = self.reliability_weighted_logits(
                cross_view_distance, pos_weight, pos_mask
            )
            partition = torch.logsumexp(
                torch.cat(
                    (positive_cross, negative_cross, negative_inter),
                    dim=1,
                ),
                dim=1,
            )
            per_anchor = -positive_mean + partition
            if not bool(valid_anchor.any().item()):
                return zi.sum() * 0.0
            return per_anchor[valid_anchor].mean()

        positive_loss = -torch.sum(pos_weight * cross_view_distance)
        negative_similarity = torch.cat((negative_inter, negative_cross), dim=1)
        negative_loss = torch.logsumexp(negative_similarity, dim=1, keepdim=False)
        negative_loss = torch.sum(negative_loss)

        return (positive_loss + negative_loss) / N

    def reliability_relation_barycenter_loss(
        self,
        view_features,
        consensus_features,
        reliability=None,
        trust_scores=None,
        temperature=0.50,
    ):
        """Distil one reliability-weighted relation barycenter into consensus.

        Every dataset, regardless of scale, evaluates exactly this ``B x B``
        objective.  A view proposes a row-stochastic relation distribution;
        sample-view reliability supplies pairwise barycentric mass.  The
        detached barycenter is the teacher and the learned consensus is the
        only student.  This avoids choosing a view or a relation backend in
        the loss itself and keeps gradients out of the pseudo-target.
        """
        if not view_features:
            raise ValueError("view_features must contain at least one view")
        batch_size = int(consensus_features.size(0))
        if batch_size < 2:
            return consensus_features.sum() * 0.0
        if any(view.shape != consensus_features.shape for view in view_features):
            raise ValueError("all relation features must have the same [B,D] shape")
        temperature = max(float(temperature), 1e-6)
        device = consensus_features.device
        dtype = consensus_features.dtype
        view_count = len(view_features)
        if reliability is None:
            reliability = torch.full(
                (batch_size, view_count),
                1.0 / float(view_count),
                device=device,
                dtype=dtype,
            )
        else:
            reliability = reliability.to(device=device, dtype=dtype)
            if reliability.shape != (batch_size, view_count):
                raise ValueError("reliability must have shape [B,V]")
            reliability = reliability.clamp_min(1e-8)
            reliability = reliability / reliability.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)

        diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)

        def relation_distribution(features):
            features = F.normalize(features.float(), dim=1)
            logits = torch.mm(features, features.t()) / temperature
            logits = logits.masked_fill(diagonal, -torch.inf)
            return F.softmax(logits, dim=1).to(dtype)

        with torch.no_grad():
            numerator = torch.zeros(
                batch_size, batch_size, device=device, dtype=dtype
            )
            denominator = torch.zeros_like(numerator)
            for view_index, features in enumerate(view_features):
                proposal = relation_distribution(features.detach())
                pair_mass = torch.sqrt(
                    reliability[:, view_index : view_index + 1]
                    * reliability[:, view_index].unsqueeze(0)
                )
                numerator.add_(pair_mass * proposal)
                denominator.add_(pair_mass)
            target = numerator / denominator.clamp_min(1e-8)
            if trust_scores is not None:
                trust = trust_scores.detach().to(device=device, dtype=dtype)
                trust = trust.clamp(0.0, 1.0)
                pair_trust = torch.sqrt(trust[:, None] * trust[None, :])
                target.mul_(0.25 + 0.75 * pair_trust)
            target.masked_fill_(diagonal, 0.0)
            target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)

        consensus_probability = relation_distribution(consensus_features)
        per_sample = -(
            target * torch.log(consensus_probability.clamp_min(1e-8))
        ).sum(dim=1)
        if trust_scores is None:
            return per_sample.mean()
        query_mass = 0.25 + 0.75 * trust_scores.to(per_sample).clamp(0.0, 1.0)
        return (per_sample * query_mass).sum() / query_mass.sum().clamp_min(1e-8)

    def leave_one_out_global_correlation_loss(
        self,
        view_features,
        reliability=None,
        redundancy_weight=1.0,
    ):
        """Align every view with a consensus that excludes that same view.

        This is a label-free, reliability-weighted Barlow-style objective.  It
        removes the self-inclusion shortcut of aligning a view to a consensus
        that already contains that view.  Weighted batch standardisation makes
        the diagonal target scale-free, while the off-diagonal penalty avoids
        a redundant/collapsed representation.
        """
        if len(view_features) < 2:
            return view_features[0].sum() * 0.0
        batch_size, feature_dim = view_features[0].shape
        if batch_size < 2:
            return view_features[0].sum() * 0.0
        if any(item.shape != (batch_size, feature_dim) for item in view_features):
            raise ValueError("all view features must share shape [B,D]")
        device = view_features[0].device
        dtype = view_features[0].dtype
        view_count = len(view_features)
        if reliability is None:
            reliability = torch.full(
                (batch_size, view_count),
                1.0 / float(view_count),
                device=device,
                dtype=dtype,
            )
        else:
            reliability = reliability.to(device=device, dtype=dtype)
            if reliability.shape != (batch_size, view_count):
                raise ValueError("reliability must have shape [B,V]")
            reliability = reliability.clamp_min(1e-6)
            reliability = reliability / reliability.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)

        def weighted_standardize(features, sample_mass):
            mass = sample_mass / sample_mass.sum().clamp_min(1e-8)
            mean = (mass[:, None] * features).sum(dim=0, keepdim=True)
            centered = features - mean
            variance = (mass[:, None] * centered.square()).sum(
                dim=0, keepdim=True
            )
            return centered / torch.sqrt(variance + 1e-4), mass

        losses = []
        stacked = torch.stack(view_features, dim=1)
        identity = torch.eye(feature_dim, dtype=torch.bool, device=device)
        for view_index, features in enumerate(view_features):
            other_mass = reliability.clone()
            other_mass[:, view_index] = 0.0
            other_mass = other_mass / other_mass.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            leave_one_out = (stacked * other_mass.unsqueeze(2)).sum(dim=1)
            sample_mass = (
                reliability[:, view_index]
                * (1.0 - reliability[:, view_index])
            ).clamp_min(1e-6)
            standardized_view, normalized_mass = weighted_standardize(
                features, sample_mass
            )
            standardized_target, _ = weighted_standardize(
                leave_one_out, sample_mass
            )
            correlation = torch.mm(
                (standardized_view * normalized_mass[:, None]).t(),
                standardized_target,
            )
            diagonal_loss = (correlation.diag() - 1.0).square().mean()
            off_diagonal_loss = correlation[~identity].square().mean()
            losses.append(
                diagonal_loss
                + float(redundancy_weight) * off_diagonal_loss
            )
        return torch.stack(losses).mean()

    def pairwise_canonical_max_correlation_loss(
        self,
        projected_views,
        reliability=None,
        regularization=1e-3,
    ):
        """Maximise rotation-invariant canonical correlation across views.

        Unlike a Barlow-style diagonal target, CCA does not require two
        modalities to encode a factor in the same coordinate.  For every
        view pair we compute weighted, centred covariances, whiten each side,
        and maximise the mean singular value of the whitened cross-covariance.
        View-specific low-rank heads keep this global objective inexpensive
        and prevent it from consuming the clustering representation directly.
        """
        if not projected_views:
            raise ValueError("projected_views must not be empty")
        if len(projected_views) < 2:
            return projected_views[0].sum() * 0.0
        batch_size, output_dim = projected_views[0].shape
        if batch_size < 2:
            return projected_views[0].sum() * 0.0
        if output_dim < 1 or any(
            item.shape != (batch_size, output_dim)
            for item in projected_views
        ):
            raise ValueError("all projected views must share shape [B,D]")
        regularization = float(regularization)
        if not regularization > 0.0:
            raise ValueError("regularization must be positive")

        device = projected_views[0].device
        dtype = projected_views[0].dtype
        view_count = len(projected_views)
        if reliability is None:
            reliability = torch.full(
                (batch_size, view_count),
                1.0 / float(view_count),
                device=device,
                dtype=dtype,
            )
        else:
            reliability = reliability.detach().to(device=device, dtype=dtype)
            if reliability.shape != (batch_size, view_count):
                raise ValueError("reliability must have shape [B,V]")
            reliability = reliability.clamp_min(1e-6)
            reliability = reliability / reliability.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)

        identity = torch.eye(output_dim, device=device, dtype=dtype)
        pair_losses = []
        for left in range(view_count):
            for right in range(left + 1, view_count):
                sample_mass = torch.sqrt(
                    reliability[:, left] * reliability[:, right]
                ).clamp_min(1e-6)
                sample_mass = sample_mass / sample_mass.sum().clamp_min(1e-8)
                x = projected_views[left]
                y = projected_views[right]
                x = x - (sample_mass[:, None] * x).sum(dim=0, keepdim=True)
                y = y - (sample_mass[:, None] * y).sum(dim=0, keepdim=True)
                weighted_x = x * torch.sqrt(sample_mass[:, None])
                weighted_y = y * torch.sqrt(sample_mass[:, None])
                covariance_x = weighted_x.t().mm(weighted_x)
                covariance_y = weighted_y.t().mm(weighted_y)
                cross_covariance = weighted_x.t().mm(weighted_y)

                # Scale the ridge by the average marginal variance so the
                # same regularisation has the same meaning on every dataset.
                ridge_x = regularization * (
                    covariance_x.diag().mean().detach().clamp_min(1e-6)
                )
                ridge_y = regularization * (
                    covariance_y.diag().mean().detach().clamp_min(1e-6)
                )
                cholesky_x = torch.linalg.cholesky(
                    covariance_x + ridge_x * identity
                )
                cholesky_y = torch.linalg.cholesky(
                    covariance_y + ridge_y * identity
                )
                whitened_cross = torch.linalg.solve_triangular(
                    cholesky_x,
                    cross_covariance,
                    upper=False,
                )
                whitened_cross = torch.linalg.solve_triangular(
                    cholesky_y,
                    whitened_cross.t(),
                    upper=False,
                ).t()
                canonical_correlations = torch.linalg.svdvals(
                    whitened_cross
                )
                pair_losses.append(1.0 - canonical_correlations.mean())
        return torch.stack(pair_losses).mean()

    def leave_one_out_instance_contrastive_loss(
        self,
        view_features,
        reliability=None,
        soft_targets=None,
        trust_scores=None,
        temperature=0.20,
        false_negative_floor=0.05,
    ):
        """Contrast each view with the same sample in the other views.

        The positive is always the matching sample and therefore does not
        depend on a pseudo label.  A reliability-weighted consensus excluding
        the query view supplies the stop-gradient target, so a dominant view
        is not pulled toward its own copy or toward a weak view.  Pseudo-label
        probabilities are used only to reduce likely false-negative mass; the
        trust score smoothly disables that masking when the pseudo target is
        uncertain.
        """
        if not view_features:
            raise ValueError("view_features must contain at least one view")
        if len(view_features) < 2:
            return view_features[0].sum() * 0.0
        batch_size, feature_dim = view_features[0].shape
        if batch_size < 2:
            return view_features[0].sum() * 0.0
        if any(item.shape != (batch_size, feature_dim) for item in view_features):
            raise ValueError("all view features must share shape [B,D]")
        temperature = float(temperature)
        false_negative_floor = float(false_negative_floor)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < false_negative_floor <= 1.0:
            raise ValueError("false_negative_floor must lie in (0,1]")

        device = view_features[0].device
        dtype = view_features[0].dtype
        view_count = len(view_features)
        if reliability is None:
            reliability = torch.full(
                (batch_size, view_count),
                1.0 / float(view_count),
                device=device,
                dtype=dtype,
            )
        else:
            reliability = reliability.detach().to(device=device, dtype=dtype)
            if reliability.shape != (batch_size, view_count):
                raise ValueError("reliability must have shape [B,V]")
            reliability = reliability.clamp_min(1e-8)
            reliability = reliability / reliability.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)

        negative_mass = torch.ones(
            batch_size, batch_size, device=device, dtype=dtype
        )
        if soft_targets is not None:
            targets = soft_targets.detach().to(device=device, dtype=dtype)
            if targets.ndim != 2 or targets.size(0) != batch_size:
                raise ValueError("soft_targets must have shape [B,K]")
            targets = targets.clamp_min(0.0)
            targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
            same_cluster_probability = torch.mm(targets, targets.t()).clamp(
                0.0, 1.0
            )
            if trust_scores is None:
                pair_trust = torch.ones_like(same_cluster_probability)
            else:
                trust = trust_scores.detach().to(device=device, dtype=dtype)
                if trust.shape != (batch_size,):
                    raise ValueError("trust_scores must have shape [B]")
                trust = trust.clamp(0.0, 1.0)
                pair_trust = torch.sqrt(trust[:, None] * trust[None, :])
            negative_mass = (
                1.0 - pair_trust * same_cluster_probability
            ).clamp(min=false_negative_floor, max=1.0)
        diagonal = torch.arange(batch_size, device=device)
        negative_mass[diagonal, diagonal] = 1.0
        log_negative_mass = torch.log(negative_mass.clamp_min(1e-8))

        stacked = torch.stack(view_features, dim=1)
        losses = []
        masses = []
        for view_index, features in enumerate(view_features):
            other_reliability = reliability.clone()
            other_reliability[:, view_index] = 0.0
            available_mass = other_reliability.sum(dim=1)
            other_reliability = other_reliability / available_mass[:, None].clamp_min(
                1e-8
            )
            leave_one_out = (
                stacked.detach() * other_reliability.unsqueeze(2)
            ).sum(dim=1)
            query = F.normalize(features.float(), dim=1)
            target = F.normalize(leave_one_out.float(), dim=1)
            logits = torch.mm(query, target.t()) / temperature
            logits = logits + log_negative_mass.float()
            per_sample = -logits.diag() + torch.logsumexp(logits, dim=1)

            # The mass of the other views is high precisely when their target
            # is trustworthy.  This makes the update directional: a weak view
            # learns from a strong leave-one-out teacher without pulling the
            # strong view equally hard toward the weak one.
            query_mass = available_mass.clamp_min(1e-6)
            losses.append(per_sample)
            masses.append(query_mass)

        loss_matrix = torch.stack(losses, dim=1)
        mass_matrix = torch.stack(masses, dim=1).to(loss_matrix)
        return (loss_matrix * mass_matrix).sum() / mass_matrix.sum().clamp_min(
            1e-8
        )

    @staticmethod
    def _symmetric_instance_nt_xent(left, right, temperature=0.5):
        """Dynamic-batch NT-Xent for corresponding samples in two views."""
        if left.shape != right.shape or left.ndim != 2:
            raise ValueError("paired projections must share shape [B,D]")
        batch_size = int(left.size(0))
        if batch_size < 2:
            return left.sum() * 0.0
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        features = F.normalize(torch.cat([left, right], dim=0), dim=1)
        logits = torch.mm(features, features.t()) / temperature
        diagonal = torch.eye(
            2 * batch_size, dtype=torch.bool, device=features.device
        )
        logits = logits.masked_fill(diagonal, -torch.inf)
        positive = torch.arange(2 * batch_size, device=features.device)
        positive = (positive + batch_size) % (2 * batch_size)
        return F.cross_entropy(logits, positive)

    def multilevel_instance_contrastive_loss(
        self, projected_views, temperature=0.5
    ):
        """Average pairwise instance contrast in a separate projection space."""
        if len(projected_views) < 2:
            return projected_views[0].sum() * 0.0
        losses = []
        for left in range(len(projected_views)):
            for right in range(left + 1, len(projected_views)):
                losses.append(
                    self._symmetric_instance_nt_xent(
                        projected_views[left],
                        projected_views[right],
                        temperature=temperature,
                    )
                )
        return torch.stack(losses).mean()

    def multilevel_cluster_contrastive_loss(
        self, view_logits, temperature=1.0
    ):
        """Align cluster columns across views while preventing empty clusters."""
        if len(view_logits) < 2:
            return view_logits[0].sum() * 0.0
        probabilities = [F.softmax(logits, dim=1) for logits in view_logits]
        pair_losses = []
        for left in range(len(probabilities)):
            for right in range(left + 1, len(probabilities)):
                pair_losses.append(
                    self.cluster_assignment_contrastive_loss(
                        view_logits[left],
                        view_logits[right],
                        temperature=temperature,
                    )
                )
        # KL(mean assignment || uniform) is the cluster-level anti-collapse
        # term used by multi-level contrastive clustering, averaged so its
        # strength does not grow with the number of views.
        balance = []
        for probability in probabilities:
            marginal = probability.mean(dim=0).clamp_min(1e-8)
            balance.append(
                torch.log(
                    torch.tensor(
                        float(marginal.numel()),
                        device=marginal.device,
                        dtype=marginal.dtype,
                    )
                )
                + torch.sum(marginal * torch.log(marginal))
            )
        return torch.stack(pair_losses).mean() + torch.stack(balance).mean()

    def shared_private_orthogonality_loss(
        self,
        shared_features,
        private_features,
    ):
        """Remove linear redundancy between shared and private branches."""
        if len(shared_features) != len(private_features) or not shared_features:
            raise ValueError("shared/private feature lists must be non-empty and aligned")
        losses = []
        for shared, private in zip(shared_features, private_features):
            if shared.shape != private.shape or shared.ndim != 2:
                raise ValueError("shared/private features must share shape [B,D]")
            if shared.size(0) < 2:
                losses.append(shared.sum() * 0.0)
                continue
            shared = shared.float() - shared.float().mean(dim=0, keepdim=True)
            private = private.float() - private.float().mean(dim=0, keepdim=True)
            shared = F.normalize(shared, dim=0)
            private = F.normalize(private, dim=0)
            cross_correlation = torch.mm(shared.t(), private)
            losses.append(cross_correlation.square().mean())
        return torch.stack(losses).mean()

    def shared_private_sample_cosine_loss(
        self,
        shared_features,
        private_features,
    ):
        """DMCAR-style squared cosine separation for each sample and view."""
        if len(shared_features) != len(private_features) or not shared_features:
            raise ValueError("shared/private feature lists must be non-empty and aligned")
        losses = []
        for shared, private in zip(shared_features, private_features):
            if shared.shape != private.shape or shared.ndim != 2:
                raise ValueError("shared/private features must share shape [B,D]")
            cosine = F.cosine_similarity(
                shared.float(), private.float(), dim=1, eps=1e-8
            )
            losses.append(cosine.square().mean())
        return torch.stack(losses).mean()

    def prototype_consistency_loss(
        self,
        logits,
        soft_targets,
        trust_scores=None,
    ):
        """
        全局软原型一致性损失。
        旧版只在固定 boundary 百分比上计算，容易只提升局部子集。
        新版没有 boundary ratio：所有样本都受到原型分配约束，低可信样本用软目标，高可信样本近似硬目标。
        """
        log_prob = F.log_softmax(logits, dim=1)
        per_sample = -(soft_targets * log_prob).sum(dim=1)

        weight = torch.ones_like(per_sample)
        if trust_scores is not None:
            # 高可信样本更强监督；低可信样本仍参与，但权重较小，不再被完全丢掉。
            weight = 0.25 + 0.75 * trust_scores.float().clamp(0.0, 1.0)

        loss = (per_sample * weight).sum() / (weight.sum() + 1e-8)
        return loss

    def assignment_alignment_loss(
        self,
        view_logits,
        target_probs,
        trust_scores=None,
        confidence_power=0.0,
        confidence_blend=1.0,
    ):
        """
        视图级簇分配一致性损失：让每个单视图的簇分配接近融合表示的全局软目标。
        作用是替代“只看 d2-d1 边界子集”的局部策略，直接优化全体样本的聚类分配。
        """
        log_prob = F.log_softmax(view_logits, dim=1)
        per_sample = -(target_probs.detach() * log_prob).sum(dim=1)
        if trust_scores is not None:
            trust = trust_scores.float().clamp(0.0, 1.0)
            confidence_power = float(confidence_power)
            confidence_blend = float(confidence_blend)
            if not 0.0 <= confidence_blend <= 1.0:
                raise ValueError("confidence_blend must lie in [0,1]")
            if confidence_power > 0.0:
                # Continuous self-paced weighting: uncertain pseudo labels are
                # not hard-filtered, but their gradients vanish smoothly.
                legacy_weight = 0.25 + 0.75 * trust
                filtered_weight = trust.pow(confidence_power)
                weight = (
                    (1.0 - confidence_blend) * legacy_weight
                    + confidence_blend * filtered_weight
                )
            else:
                weight = 0.25 + 0.75 * trust
            return (per_sample * weight).sum() / (weight.sum() + 1e-8)
        return per_sample.mean()

    def cluster_balance_loss(self, logits):
        """
        轻量簇均衡正则，思想来自 IIC/SwAV 一类方法中防止全部样本塌缩到少数簇的设计。
        这里只是很小的辅助项，不能过大；过大可能伤害真实类别不均衡的数据集。
        """
        p = F.softmax(logits, dim=1)
        avg_p = p.mean(dim=0)
        uniform = torch.full_like(avg_p, 1.0 / avg_p.numel())
        return torch.sum(avg_p * (torch.log(avg_p + 1e-8) - torch.log(uniform + 1e-8)))

    def target_sharpen_loss(self, logits, trust_scores=None):
        """
        轻量置信度增强：鼓励样本分配不要过分平均，但用很小权重，避免强行硬化边界样本。
        """
        p = F.softmax(logits, dim=1)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)
        if trust_scores is not None:
            weight = 0.25 + 0.75 * trust_scores.float().clamp(0.0, 1.0)
            return (entropy * weight).sum() / (weight.sum() + 1e-8)
        return entropy.mean()


    def cluster_assignment_contrastive_loss(self, logits_a, logits_b, temperature=0.5):
        """
        BGCR-v38：簇分配级对比损失。
        思想来自 AAAI 2021 Contrastive Clustering：把 soft assignment 的列看作簇表示，
        让同一簇在不同视图/融合表示中的分配列相互靠近，不同簇相互分开。
        作用：直接提高 Full dataset 的簇级判别性，避免只改少数边界样本。
        """
        qa = F.softmax(logits_a, dim=1)
        qb = F.softmax(logits_b, dim=1)
        ca = F.normalize(qa.t(), dim=1)
        cb = F.normalize(qb.t(), dim=1)
        logits = torch.mm(ca, cb.t()) / max(float(temperature), 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def compute_cluster_loss(self, q_centers, k_centers, psedo_labels):
        d_q = q_centers.mm(q_centers.T) / self.temperature_l
        d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        d_q = d_q.float()
        d_q[torch.arange(self.num_clusters, device=d_q.device), torch.arange(self.num_clusters, device=d_q.device)] = d_k

        if psedo_labels.ndim > 1:
            class_count = psedo_labels.float().sum(dim=0).to(d_q.device)
        else:
            class_count = torch.bincount(
                psedo_labels.long(), minlength=self.num_clusters
            ).to(d_q.device, dtype=d_q.dtype)
        zero_classes = torch.arange(self.num_clusters, device=d_q.device)[
            class_count <= 0
        ]
        mask = torch.zeros((self.num_clusters, self.num_clusters), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.num_clusters, self.num_clusters), device=d_q.device)
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.num_clusters - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.num_clusters, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        # A K=100, B=256 batch contains many singleton pseudo-classes.  Their
        # centres are noisy individual samples and should not carry the same
        # mass as well-supported centres.  Occupancy weighting is dataset
        # independent and converges to the original uniform class loss when
        # batch occupancy is balanced.
        if self.cluster_occupancy_weighting:
            positive_count = class_count[class_count > 0]
            mean_count = positive_count.mean().clamp_min(1.0)
            occupancy = (class_count / mean_count).clamp(
                min=0.0, max=1.0
            )
            loss = (loss * occupancy).sum() / occupancy.sum().clamp_min(1e-8)
        else:
            loss = loss.sum() / (
                self.num_clusters - len(zero_classes) + 1e-8
            )
        return loss
