function V = ind2vec(ind)
% Local compatibility function, not a MathWorks toolbox implementation.
% Supports only the finite positive integer label-vector signature used by GDR.
assert(isvector(ind) && ~isempty(ind), 'GDRCompat:ind2vec:shape');
ind = double(ind(:)');
assert(all(isfinite(ind)) && all(ind >= 1) && all(ind == fix(ind)), ...
    'GDRCompat:ind2vec:labels');
V = sparse(ind, 1:numel(ind), 1, max(ind), numel(ind));
% Optional read-only observation of the first indicator construction in the
% unchanged author main function. This copies values and never changes them.
global GDR_CAPTURE;
if isstruct(GDR_CAPTURE) && isfield(GDR_CAPTURE, 'enabled') && GDR_CAPTURE.enabled
    stk = dbstack;
    if numel(stk) >= 2 && strcmp(stk(2).name, 'MvC_CGDR')
        names = {'X','Z','ZZ','B','A','WA','alpha','g','k','h','n','m','c'};
        GDR_CAPTURE.enabled = false;
        for i = 1:numel(names)
            GDR_CAPTURE.(names{i}) = evalin('caller', names{i});
        end
    end
end
end
