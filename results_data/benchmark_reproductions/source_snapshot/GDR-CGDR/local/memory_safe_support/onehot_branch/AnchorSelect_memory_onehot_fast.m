function [Z,ZZ,XX,m,varargout] = AnchorSelect_memory_onehot_fast(X,n,n_view,q)
% Memory-bounded AnchorSelect with corrected StartInd-to-onehot mapping.
% The only intentional semantic change relative to the retained release is
% that a sampled class2 initializes column2, rather than column1. This is a
% correction to released-code mapping semantics, not a claim about a paper
% requirement. No empty-cluster repair, seed changes, or score-based branch.
% Do not use the legacy all-first-column cross-seed anchor cache here.
% Distance blocks, source eps denominator, ten iterations, sorting/cp rules,
% random call boundaries and returned pre-update iteration10 centers remain.
% XX=[] because supported production callers discard the joined input.
% Fixed resource threshold: nodes n<=2048 delegate to unchanged author hKM.
% Its TransformL must be the independently corrected class-mapping shim.
% Optional fifth output replays only small delegated subtrees for diagnostics.
% Production four-output calls execute no such diagnostic replay.

assert(isequal(TransformL([1;2],2),eye(2)), 'GDROnehotFast:CorrectShimRequired');
assert(iscell(X) && numel(X)==n_view && q>=1 && q==fix(q));
assert(all(cellfun(@(a) size(a,1)==n && isa(a,'double'),X)));
m = 2^q;
Z = cell(1,n_view);
for v=1:n_view
    Z{v}=zeros(m,size(X{v},2));
end
XX=[];
tracing=nargout>4;
trace=struct('nodes',{{}},'root_output_labels',zeros(n,1), ...
    'feature_block_columns',256,'node_count',0, ...
    'native_hKM_threshold',2048,'delegated_subtrees',0,'blocked_nodes',0);
visit(1:n,q,1);
ZZ=[Z{:}];
if tracing
    trace.node_count=numel(trace.nodes);
    varargout{1}=trace;
end

    function visit(idx,k,first_anchor)
        if numel(idx)<=2048
            % Only this bounded node is joined; never a large root array.
            local=[];
            for vv=1:n_view;local=[local X{vv}(idx,:)];end
            before=rng;
            [ys,Cnative]=hKM(local',1:numel(idx),k,1);
            after=rng;
            col1=1;rows=first_anchor:first_anchor+2^k-1;
            for vv=1:n_view
                col2=col1+size(X{vv},2)-1;
                Z{vv}(rows,:)=Cnative(col1:col2,:)';
                col1=col2+1;
            end
            trace.delegated_subtrees=trace.delegated_subtrees+1;
            if tracing
                % Trace transcription is separately gated against real hKM.
                % Reset/restore RNG so diagnostic replay is observational.
                rng(before);
                [~,diagZZ,~,~,subtrace]=AnchorSelect_reference_trace({local},numel(idx),1,k);
                assert(isequal(diagZZ,Cnative')&& ...
                    isequal(subtrace.root_output_labels,ys(:))&&isequal(rng,after), ...
                    'GDROnehotFast:TraceReplayMismatch');
                for jj=1:numel(subtrace.nodes)
                    node=subtrace.nodes{jj};node.idx=idx(node.idx);node.idx=node.idx(:);
                    node.first_anchor=first_anchor+node.first_anchor-1;
                    trace.nodes{end+1}=node;
                end
                trace.root_output_labels(idx)=ys(:);
                rng(after);
            end
            return;
        end
        trace.blocked_nodes=trace.blocked_nodes+1;
        if k==1
            ratio=0.0;
        else
            ratio=0.5;
        end
        [C,y,node]=balanced(idx,ratio);
        if tracing
            node.idx=idx(:); node.k=k;
            node.first_anchor=first_anchor;
            trace.nodes{end+1}=node;
        end
        if k>1
            id1=find(y==1); id2=find(y==2);
            visit(idx(id1),k-1,first_anchor);
            visit(idx(id2),k-1,first_anchor+2^(k-1));
        else
            for vv=1:n_view
                Z{vv}(first_anchor:first_anchor+1,:)=C{vv}';
            end
            if tracing
                % Author hKM resets count=1 at every node, hence 3-y.
                trace.root_output_labels(idx)=3-y;
            end
        end
    end

    function [C,y,node]=balanced(idx,ratio)
        node_n=numel(idx); class_num=2;
        % Retain exactly one source random draw; correct only its class mapping.
        StartInd=randsrc(node_n,1,1:class_num);
        F=zeros(node_n,class_num);
        F(sub2ind([node_n,class_num],(1:node_n)',StartInd))=1;
        if ratio>0.5; error('ratio should not larger than 0.5'); end
        if ratio<0; ratio=0; end
        a=floor(node_n*ratio); b=floor(node_n*(1-ratio));
        C=cell(1,n_view);
        for vv=1:n_view; C{vv}=zeros(size(X{vv},2),2); end
        node=struct();
        if tracing
            node.initial_labels=StartInd;
            node.cp_history=zeros(1,10);
            node.y_history=zeros(node_n,10);
            node.sort_history=zeros(node_n,10);
            node.q_history=zeros(node_n,10);
        end
        for iter=1:10
            % This is sum(F+eps), not sum(F)+eps: retain author behavior.
            denominator=sum(F+eps);
            aa=zeros(node_n,1); bb=zeros(1,2); ab=zeros(node_n,2);
            for vv=1:n_view
                dv=size(X{vv},2);
                for c1=1:256:dv
                    cols=c1:min(c1+255,dv);
                    block=X{vv}(idx,cols);
                    centers=(block'*F)./denominator;
                    C{vv}(cols,:)=centers;
                    aa=aa+sum(block.*block,2);
                    bb=bb+sum(centers.*centers,1);
                    ab=ab+block*centers;
                end
            end
            F=sparse(node_n,class_num);
            Q=bsxfun(@plus,aa,bb)-2*ab;
            Q(Q<0)=0;
            distances=Q(:,1)-Q(:,2);
            [temp,order]=sort(distances);
            nn=length(find(temp<0));
            if nn>=a && nn<=b; cp=nn;
            elseif nn<a; cp=a;
            else; cp=b;
            end
            if cp<1; cp=1;
            elseif cp>node_n-1; cp=node_n-1;
            end
            F(order(1:cp),1)=1;
            F(:,2)=1-F(:,1);
            if tracing
                [~,current_y]=max(F,[],2);
                node.cp_history(iter)=cp;
                node.y_history(:,iter)=current_y;
                node.sort_history(:,iter)=order;
                node.q_history(:,iter)=distances;
            end
        end
        [~,y]=max(F,[],2);
        % Return the centers computed before iteration 10's F update, just
        % like the source BalancedKM. Do not recompute centers from final F.
    end
end
