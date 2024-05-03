function y_pred = idpredict(model, z, horizon)

    zero = max(model.na, model.nb+model.nk-1);
    z = [zeros(zero,2);z];

    phi = uy2phi(z,[model.na, model.nb, model.nk]); % Generate Regressor
    phi(:,1) = [];
    y_pred = phi(:,1);

    for k = 1:horizon

        phi(:,k) = y_pred;
        y_pred = phi*model.theta;
        

    end

    
end

