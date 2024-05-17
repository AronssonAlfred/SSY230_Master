function oe_model = oefit(z, n)
    na = n(1);
    nb = n(2);
    nk = n(3);

    % High-order ARX model estimation
    high_order = [4*na, 4*nb, nk];  % Ensuring nf is not less than nb

    arx_model = arxfit(z, high_order);

    % Simulating ARX model output
    u = z(:, 2);
    ys = idsimulate(arx_model, u);

    % OE model estimation using original order parameters
    oe_model = arxfit([ys, u], [na, nb, nk]);
    oe_model.type = 'OE';
    oe_model.na = na;
    oe_model.nb = nb;
    oe_model.nk = nk;

    oe_model.F = oe_model.theta(1:na);
    oe_model.B = oe_model.theta(nb+1:end);
end





