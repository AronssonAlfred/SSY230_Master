function oe_model = oefit(z, n)
    nb = n(1);  % Number of B coefficients
    nf = n(2);  % Number of F coefficients
    nk = n(3);  % Input delay

    % Estimate a high-order ARX model
    high_order = [4*nf, 4*nb, nk];
    arx_model = arxfit(z, high_order);
%     arx_model.na = nf;
%     arx_model.nb = nb;


    % Simulate the ARX model output
    u = z(:, 2);  % input data
    ys = idsimulate(arx_model, u);

    % Estimate the OE model using the original model order
    oe_model = arxfit([ys, u], [nf, nb, nk]);  % Fit as ARX using output from ARX simulation
    oe_model.type = 'OE';  
    oe_model.nf = nf;
    oe_model.nb = nb;
    oe_model.nk = nk;
    
    oe_model.B = oe_model.theta(1:nb)';
    oe_model.F = oe_model.theta(nb+1:nb+nk);

end