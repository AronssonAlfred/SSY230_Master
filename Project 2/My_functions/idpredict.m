function y_pred = idpredict(model, z, horizon)


    if isequals(modell.type,'ARX')

        zero = max(model.na, model.nb + model.nk - 1); % Extend model order for zero padding for t < 0
        phi = uy2phi([zeros(zero, 2); z], [model.na, model.nb, model.nk]); % Generate regressor
        phi(:, 1) = []; % Remove the first column
        y_pred = phi * model.theta; % 1 step prediction

        for k = 2:horizon

            phi = [[zeros(k,model.na); y_pred(k:end-1), phi(k:length(phi)-1,1:model.na-1)], [phi(:,model.na+1:end)]];
            y_pred = phi * model.theta; % 1 step prediction

        end

    end

    if isequals(modell.type,'OE')

        zero = max(model.nf, model.nb + model.nk - 1); % Extend model order for zero padding for t < 0
        phi = uy2phi([zeros(zero, 2); z], [model.na, model.nb, model.nk]); % Generate regressor
        phi(:, 1) = []; % Remove the first column
        y_pred = phi * model.theta; % 1 step prediction

        for k = 2:horizon

            phi = [[zeros(k,model.nf); y_pred(k:end-1), phi(k:length(phi)-1,1:model.na-1)], [phi(:,model.na+1:end)]];
            y_pred = phi * model.theta; % 1 step prediction

        end

    end

    
end
