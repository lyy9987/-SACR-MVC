function R = randsample(n, k)
% Local compatibility function, not a MathWorks toolbox implementation.
% Only uniform sampling without replacement from 1:n, as used in litekmeans.
assert(nargin == 2 && isscalar(n) && isscalar(k) && ...
    n >= 1 && k >= 0 && k <= n && n == fix(n) && k == fix(k), ...
    'GDRCompat:randsample:signature');
R = randperm(n, k)';
end
