function [x2,allpot] = poly_x2(x,n)

[Nx,nx] = size(x);

for i=1:Nx
    allComb = arrayfun(@(x) 0:n, 1:nx, 'UniformOutput', false);
    [allComb{:}] = ndgrid(allComb{:});
    allComb = cell2mat(cellfun(@(x) x(:), allComb, 'UniformOutput', false));
    allpot = allComb(sum(allComb, 2) <= n, :);
end
nmon=size(allpot,1);

x2 = zeros(Nx,nmon);

for ii = 1:nmon
    monomial = ones(Nx, 1);
    for jj = 1:nx
        monomial = monomial .* x(:, jj).^allpot(ii, jj);
    end
    x2(:, ii) = monomial;
end
end