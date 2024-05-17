function idcompare(model, z, horizon)
    % Define variables needed
    t = (1:length(z))';
    y_pred = idpredict(model, z, horizon);
    y_sim = idsimulate(model, z(:,2));
    n_samples = 5000; % Number of samples for the Monte Carlo simulation
    d = length(model.theta); % Dimension of the parameter space
    confidence_level = 0.95; % Typically for a 95% confidence interval
    chi_sq_value = chi2inv(confidence_level, d); % Chi-squared value for the confidence level

    figure;% Initialize subplots

    % Prediction subplot
    subplot(2,1,1)
    plot(t, z(:,1), 'DisplayName', 'Measured y') % Plot measured y
    hold on
    plot(t, y_pred, 'DisplayName', 'Predicted y'); % Plot predicted y

    % Check if variance is available for uncertainty calculation
    if isfield(model, 'variance') && ~isempty(model.variance)
        % Uncertainty algorithm for prediction
        % Preallocate arrays to store parameter samples and function evaluations
        f_evals = zeros(length(z), n_samples);

        % Generate parameter samples and corresponding function evaluations
        for i = 1:n_samples
            Delta_theta = randn(d, 1);
            Delta_theta = Delta_theta / norm(Delta_theta) * sqrt(chi_sq_value);
            theta_k = model.theta + sqrtm(model.variance) * Delta_theta; % Adjust the estimated parameter
            model_sample = model;
            model_sample.theta = theta_k;
            f_evals(:, i) = idpredict(model_sample, z, horizon); % Evaluate the function for the k-th sample
        end

        % Calculate upper and lower bounds of the function evaluations
        f_max = max(f_evals, [], 2);
        f_min = min(f_evals, [], 2);
        plot(t, f_min, 'b', 'DisplayName', 'Lower Bound')
        plot(t, f_max, 'r', 'DisplayName', 'Upper Bound')
        Xr = [t; flipud(t)]; % Create a column vector for x coordinates
        Y = [f_min; flipud(f_max)]; % Concatenate y_min with flipped y_max
        fill(Xr, Y, 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Fill with green color and 10% opacity
    else
        warning('Model variance not specified. Uncertainty analysis will be skipped.');
    end

    title('Prediction')
    legend show
    hold off

    % Simulation plot
    subplot(2,1,2)
    plot(t, z(:,1), 'DisplayName', 'Measured y') % Plot measured y
    hold on
    plot(t, y_sim, 'DisplayName', 'Simulated y') % Plot simulated y

    % Check if variance is available for uncertainty calculation
    if isfield(model, 'variance') && ~isempty(model.variance)
        % Uncertainty algorithm for Simulation
        % Preallocate arrays to store parameter samples and function evaluations
        f_evals = zeros(length(z), n_samples);

        % Generate parameter samples and corresponding function evaluations
        for i = 1:n_samples
            Delta_theta = randn(d, 1);
            Delta_theta = Delta_theta / norm(Delta_theta) * sqrt(chi_sq_value);
            theta_k = model.theta + sqrtm(model.variance) * Delta_theta; % Adjust the estimated parameter
            model_sample = model;
            model_sample.theta = theta_k;
            f_evals(:, i) = idsimulate(model_sample, z(:,2)); % Evaluate the function for the k-th sample
        end

        % Calculate upper and lower bounds of the function evaluations
        f_max = max(f_evals, [], 2);
        f_min = min(f_evals, [], 2);
        plot(t, f_min, 'b', 'DisplayName', 'Lower Bound')
        plot(t, f_max, 'r', 'DisplayName', 'Upper Bound')
        Xr = [t; flipud(t)];
        Y = [f_min; flipud(f_max)];
        fill(Xr, Y, 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Fill with green color and 10% opacity
    else
        warning('Model variance not specified. Uncertainty analysis will be skipped.');
    end

    hold off
    title('Simulation')
    legend show
end
