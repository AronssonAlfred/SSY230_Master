function m = LinRegress(x,y)

    [Ny,ny] = size(y);
    [Nx,nx] = size(x);
    m.model = 'LR'; % indicate that it is a linear regression model
    
    if Nx == Ny % Test that the number of data is the same in x and y
        
        m.theta = (x'*x)\x'*y;    
        
        if ny == 1 % calculate uncertainty only in the case that there is one output
            
           residuals = y - evalModel(m, x);
           V = 1/2*residuals'*(eye(Nx)-x*(x'*x)^-1*x')*residuals;
           lambda_squared = 2*V/(Nx-nx);
           m.variance = lambda_squared*(x'*x)^-1;
           
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