function plotModel(x,y,varargin)



[Nx,nx] = size(x);
X = x;

t = (0:length(x)-1)';

for m = 1:length(varargin)
    model = varargin{m};

    if isequal(model.model,'LR') || isequal(model.model,'POLY')

        if nx == 1

            plot(t,y)
            hold on


            if length(varargin)==1

                % Assuming you have `theta_hat` (your estimated parameter vector) and `P` (the covariance matrix of `theta_hat`)
                n_samples = 20; % Number of samples for the Monte Carlo simulation
                d = length(model.theta); % Dimension of the parameter space
                confidence_level = 0.95; % Typically for a 95% confidence interval
                chi_sq_value = chi2inv(confidence_level, d); % Chi-squared value for the confidence level

                % Preallocate arrays to store parameter samples and function evaluations
                f_evals = zeros(length(X), n_samples); % Assuming x is a vector of input features

                % Generate parameter samples and corresponding function evaluations
                for i = 1:n_samples
                    Delta_theta = randn(d, 1);
                    Delta_theta = Delta_theta / norm(Delta_theta) * sqrt(chi_sq_value);
                    theta_k = model.theta + sqrtm(model.variance) * Delta_theta; % Adjust the estimated parameter
                    model_sample = model;
                    model_sample.theta = theta_k;

                    phi = uy2phi([y x], [model.na, model.nb, model.nb]);
                    phi(:,1) = [];
                    model_sample.phi = phi;
                    

                    f = zeros(length(y),1);
                    f(end-length(model.phi)+1:end) = model_sample.phi*model_sample.theta;

                    f_evals(:, i) = f; % Evaluate the function for the k-th sample
                end

                % Calculate upper and lower bounds of the function evaluations
                f_max = max(f_evals, [], 2);
                f_min = min(f_evals, [], 2);

                plot(t,f_min,'b')
                plot(t,f_max,'r')
                Xr = [t; flipud(t)]; % Note the semicolon to create a column vector
                Y = [f_min; flipud(f_max)]; % Concatenate y_min with flipped y_max
   
                fill(Xr, Y, 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Fill with green color and 10% opacity
             end

        else
            disp('Can only plot for dimensions 1 and 2')
        end

    end

    

end
end