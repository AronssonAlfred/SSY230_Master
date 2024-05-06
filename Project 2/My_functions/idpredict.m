function y_pred = idpredict(model, z, horizon)

    zero = max(model.na, model.nb + model.nk - 1);
    phi = uy2phi([zeros(zero, 2); z], [model.na, model.nb, model.nk]); % Generate regressor
    phi(:, 1) = []; % Remove the first column (if needed, depending on your uy2phi implementation)
    y_pred = phi * model.theta; % Initial prediction


    for k = 2:horizon
        
        zero = max(zero, k);
        phi_temp = uy2phi([zeros(zero, 2); [y_pred, z(:,2)]], [model.na, model.nb, model.nk]);
        phi_temp = phi_temp(1:length(phi),:);
        phi(:,1:horizon-2) = phi_temp(:,1:horizon-2);
        y_pred = phi * model.theta;

    end

end
