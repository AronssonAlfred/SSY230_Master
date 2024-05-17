function y_pred = evalModel(model, x)

[Nx,~] = size(x);

% Seems to work fine
if isequal(model.model,'LR')

    y_pred = x*model.theta;

% Seems to not work fine
elseif isequal(model.model, 'KNN')
    % For each input regressor, find the k closest regressors in the KNN model
    y_pred = zeros(Nx, 1);

    for i = 1:Nx
        % Calculate the distance from the i-th input regressor to all training regressors
        distances = sqrt(sum((model.x - x(i, :)).^2, 2));

        % Sort the distances and get the indices of the k smallest distances
        [~, sortedIndices] = sort(distances);

        % Select the k nearest neighbors
        nearestNeighbours = sortedIndices(1:model.k);

        % Calculate the mean of the corresponding outputs
        y_pred(i) = mean(model.y(nearestNeighbours));
    end
elseif isequal(model.model,'POLY')

    y_pred = poly_x2(x,model.n) * model.theta;

else
    error('Unknown model type');
end


end
