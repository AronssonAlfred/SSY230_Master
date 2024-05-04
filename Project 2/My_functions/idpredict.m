function y_pred = idpredict(model, z, horizon)

    zero = max(model.na, model.nb + model.nk - 1);
    phi = uy2phi([zeros(zero, 2); z], [model.na, model.nb, model.nk]); % Generate regressor
    phi(:, 1) = []; % Remove the first column (if needed, depending on your uy2phi implementation)
    y_pred = phi * model.theta; % Initial prediction

    if horizon > 1
        z = [y_pred, z(:, 2)];
        y_pred = idpredict(model, z, horizon-1);
    end
end
