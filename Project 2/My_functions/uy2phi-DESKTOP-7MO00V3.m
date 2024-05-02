function phi = uy2phi(data, shifts)

    y = data(:, 1); % First column as output y
    u = data(:, 2); % Second column as input u

    ny = shifts(1); % Number of past outputs of y to include
    nu = shifts(2); % Number of past inputs of u to include
    tau = shifts(3); % Time delay for inputs u

    N = length(y); % Total number of data points
  

    rows = min(N-ny, N-nu-tau+1);
    
    columns = 1+ny+nu;

    phi = zeros(rows,columns);

    phi(:,1) = y(N-rows+1:end);

    for i=1:ny
        phi(:,1+i) = -y(end-i-rows+1:end-i);
    end
    
    for j=1:nu
        phi(:,1+i+j) = u(end-j-rows+1-tau+1:end-j-tau+1);
    end

end
