function InitF = TransformL(F,c)
% Independent semantic correction of the released label-to-one-hot helper.
% It encodes the supplied class of each sample in that class's column.
% No RNG calls, forced nonempty clusters, or other algorithm changes.
assert(isvector(F) && isscalar(c) && c>=1 && c==fix(c), ...
    'GDRCorrectOnehot:signature');
F=double(F(:));n=numel(F);
assert(all(isfinite(F)) && all(F>=1 & F<=c & F==fix(F)), ...
    'GDRCorrectOnehot:labels');
InitF=zeros(n,c);
InitF(sub2ind([n,c],(1:n)',F))=1;
end
