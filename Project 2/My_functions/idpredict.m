function y_pred = idpredict(model, z, horizon)



    zero = max(model.na,model.nb+model.nk-1);
    z = [zeros(zero,2); z];
    

    for k = 1:horizon

        phi = uy2phi(z,[model.na, model.nb, model.nk]); % Generate Regressor

        

        phi(:,1) = [];

     

        y_pred = phi*model.theta;
        
        z(:,1) = y_pred; % Prediction as base for future predictions
        % is u also soppused to be uptaded? How?
        % y_pred is wrong size for horizon = 1 case, should be 1 longer
        % than phi, how?
        

    end

    y_pred(1:zero) = [];


    
end

