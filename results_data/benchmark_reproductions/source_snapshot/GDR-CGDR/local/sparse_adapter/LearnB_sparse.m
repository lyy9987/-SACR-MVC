function B = LearnB_sparse(B,X,Z,k,n,m,n_view)
% Resource adaptation of original LearnB: identical row-wise distance/sort/
% weights, distance blocks and sparse storage. No change to k or graph model.
block_size=256;
for v=1:n_view
    I=zeros(n*(k+1),1); J=I; S=I;
    for st=1:block_size:n
        rows=st:min(st+block_size-1,n);
        Dis=EuDist2(X{v}(rows,:),Z{v},0);
        [~,idx_m]=sort(Dis,2);
        for r=1:numel(rows)
            j=rows(r); id=idx_m(r,1:k+1); di=Dis(r,id);
            weights=(di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
            target=(j-1)*(k+1)+(1:k+1);
            I(target)=j; J(target)=id; S(target)=weights;
        end
    end
    B{v}=sparse(I,J,S,n,m);
end
end
