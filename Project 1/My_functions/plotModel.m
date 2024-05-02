function plotModel(x,y,varargin)



[Nx,nx] = size(x);
X = x;

if sum(x(:,1)) == Nx
    X(:,1) = [];
    nx = nx - 1;
end



scatter(X,y) % Plots points
hold on


for m = 1:length(varargin)
    model = varargin{m};

    if isequal(model.model,'LR') || isequal(model.model,'POLY')

        if nx == 1

            if length(model.theta)==1
                f = @(x) polyval(fliplr(([0; model.theta])'),x);
                fplot(f,[X(1), X(end)]);
            else

                f = @(x) polyval(fliplr((model.theta)'),x);
                fplot(f,[X(1), X(end)]);
            end


%             if length(varargin)==1
% 
%                 % Assuming you have `theta_hat` (your estimated parameter vector) and `P` (the covariance matrix of `theta_hat`)
%                 n_samples = 20; % Number of samples for the Monte Carlo simulation
%                 d = length(model.theta); % Dimension of the parameter space
%                 confidence_level = 0.95; % Typically for a 95% confidence interval
%                 chi_sq_value = chi2inv(confidence_level, d); % Chi-squared value for the confidence level
% 
%                 % Preallocate arrays to store parameter samples and function evaluations
%                 f_evals = zeros(length(X), n_samples); % Assuming x is a vector of input features
% 
%                 % Generate parameter samples and corresponding function evaluations
%                 for i = 1:n_samples
%                     Delta_theta = randn(d, 1);
%                     Delta_theta = Delta_theta / norm(Delta_theta) * sqrt(chi_sq_value);
%                     theta_k = model.theta + sqrtm(model.variance) * Delta_theta; % Adjust the estimated parameter
%                     model_sample = model;
%                     model_sample.theta = theta_k;
%                     f_evals(:, i) = evalModel(model_sample, x); % Evaluate the function for the k-th sample
%                 end
% 
%                 % Calculate upper and lower bounds of the function evaluations
%                 f_max = max(f_evals, [], 2);
%                 f_min = min(f_evals, [], 2);
% 
%                 plot(X,f_min,'b')
%                 plot(X,f_max,'r')
%                 Xr = [X; flipud(X)]; % Note the semicolon to create a column vector
%                 Y = [f_min; flipud(f_max)]; % Concatenate y_min with flipped y_max
%                 fill(Xr, Y, 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Fill with green color and 10% opacity
%              end

        else
            disp('Can only plot for dimensions 1 and 2')
        end

    end

    if isequal(model.model,'KNN')


        if nx == 1



            stairs(X,evalModel(model,X))


        end




    end




end
end


