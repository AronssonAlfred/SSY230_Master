function y_pred = idpredict(model, z, horizon)

    zero = max(model.na, model.nb + model.nk - 1);
    phi = uy2phi([zeros(zero, 2); z], [model.na, model.nb, model.nk]); % Generate regressor
    phi(:, 1) = []; % Remove the first column 
    y_pred = phi * model.theta; % Initial prediction


    for k = 2:horizon
        
        % Calculate y_hat(t|t-k)
        phi_temp = uy2phi([zeros(zero, 2); [y_pred, z(:,2)]], [model.na, model.nb, model.nk]);
        phi_temp(:, 1) = []; % Remove the first column 
        phi_temp = phi_temp(1:length(phi),:);
        if k <= model.na
            phi(:,1:k-1) = phi_temp(:,1:k-1);
        else
            phi(:,1:model.na) = phi_temp(:,1:model.na);
        end
        y_pred = phi * model.theta;

    end



end
