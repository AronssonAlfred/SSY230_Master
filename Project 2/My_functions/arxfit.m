function model = arxfit(z, n)

    na = n(1);
    nb = n(2);
    nk = n(3);
    % Generate the regressor matrix using uy2phi
    phi = uy2phi(z, n);
    
    % Output vector (remove first na samples)
    y = phi(:,1);
    phi(:,1) = [];

    % Estimate the ARX model parameters using linear regression
    lrmodel = LinRegress(phi, y);
    
    % Store the results in a structured array
    model.theta = [lrmodel.theta];
    model.model = lrmodel.model;
    model.variance = lrmodel.variance;
    model.type = 'ARX';
    model.na = na;
    model.nb = nb;
    model.nk = nk;
    model.phi = phi;
end