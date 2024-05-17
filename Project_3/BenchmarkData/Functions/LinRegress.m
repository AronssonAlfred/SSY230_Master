function m = LinRegress(x,y)
    [Ny,ny] = size(y);
    [Nx,nx] = size(x);
    m.model = 'LR'; % linear regr.model
    
    if Nx == Ny % Test that the number of data is the same in x and y
        
        m.theta = (x'*x)\(x'*y);
           
        if ny == 1 % calculate uncertainty only in the case that there is one output
            
            % Estimating the variance of the error term
            residuals = y - x * m.theta; %error terms for each observation, indicating how much the model's 
                                         % prediction deviates from the actual observed value

            sigma2 = (residuals' * residuals) / (Ny - nx); % according to 4.13 the residuals should be
                                                           % times 2, however this leads to high deviations 
                                                           % between the estimated and true covariance 
            % variance of the parameters
            m.variance = sigma2 * inv(x' * x); % covariance of theta_hat
            
        else
           m.variance = 'None';
        end
    else
        m.theta = NaN;
        m.model = 'None';
        m.variance = 'None';
        disp('Dimension mismatch for output and regressor.')
    end
end