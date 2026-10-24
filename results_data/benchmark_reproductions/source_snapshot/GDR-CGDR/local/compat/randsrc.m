function R = randsrc(m, n, alphabet)
% Local compatibility function, not a MathWorks toolbox implementation.
% Only uniform sampling from the finite vector used in author BalancedKM.
assert(nargin == 3 && isvector(alphabet) && ~isempty(alphabet), ...
    'GDRCompat:randsrc:signature');
assert(isscalar(m) && isscalar(n) && m >= 0 && n >= 0 && ...
    m == fix(m) && n == fix(n), 'GDRCompat:randsrc:shape');
alphabet = alphabet(:)';
R = alphabet(randi(numel(alphabet), m, n));
R = reshape(R, m, n);
end
