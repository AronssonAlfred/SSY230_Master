function y_pred = idpredict(model, z, horizon)

    zero = max(model.na, model.nb + model.nk - 1); % Extend model order for zero padding for t < 0
    phi = uy2phi([zeros(zero, 2); z], [model.na, model.nb, model.nk]); % Generate regressor
    phi(:, 1) = []; % Remove the first column 
    y_pred = phi * model.theta; % 1 step prediction

    y_regressors{1} = phi(:,1:model.na);

    for k = 2:horizon % k-step prediction
        %zero = max(k,zero);

        % Generate regressor for time t-k
        phi_temp = uy2phi([zeros(zero, 2); [y_pred, z(:, 2)]], [model.na, model.nb, model.nk]);
        phi_temp(:, 1) = [];
        phi_temp = phi_temp(1:length(phi),:);

        y_regressors{k} = phi_temp(:,1:model.na);
        
        phip = [];
        for phii = length(y_regressors)-model.na+1:length(y_regressors)
            
            %phii
            if phii <= model.na
                phip = [y_regressors{phii}(:,end-phii+1), phip];
            else
                phip = [y_regressors{phii}(:,model.na), phip];

            end

        end

        phip = [phip, phi(:,model.na+1:end)];

        % Predict y(t-k) using updated phi and model parameters
        y_pred = phip * model.theta;

    end



end
