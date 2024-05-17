function m = polyfit(x, y, lambda, n)
    % extended regressor matrix -> lambda to ->reg. param to control degree
    % of shrinkage
    [x2, allpot] = poly_x2(x, n);

    % regularized linear regression using the extended matrix
    m = LinRegressRegul(x2, y, lambda);
    m.n = n;
    m.model = 'POLY'; % polynomial regr.model
end