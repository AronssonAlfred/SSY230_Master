function y_pred = idpredict(model, z, horizon)
    zero = max(model.na, model.nb + model.nk - 1);
    phi = uy2phi([zeros(zero, 2); z], [model.na, model.nb, model.nk]); % Generate regressor
    phi(:, 1) = []; % Remove the first column (if needed, depending on your uy2phi implementation)
    y_pred = phi * model.theta; % Initial prediction

    % If more steps are needed, recursively predict further
    if horizon > 1
        % Update input data for recursive prediction
        % Assumes the model inputs are in the 2nd column of z and outputs are in the 1st column
        new_input = [y_pred, z(:, 2)];
        y_pred = idpredict(model, new_input, horizon-1);
    end
end
