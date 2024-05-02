function y_pred = idpredict(model, z, horizon)


    y_pred = zeros(length(z),1);

    for k = 1:horizon

        phi = uy2phi(z,[model.na, model.nb, model.nk]); % Generate Regressor

        phi(:,1) = []; % Why the fuck is that even there?

         y_pred(end-length(phi)+1:end) = phi*model.theta;
        
        z(:,1) = y_pred; % Prediction as base for future predictions
        % is u also soppused to be uptaded? How?
        % y_pred is wrong size for horizon = 1 case, should be 1 longer
        % than phi, how?

    end

    
end

