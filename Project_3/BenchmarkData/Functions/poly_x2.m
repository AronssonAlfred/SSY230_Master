function [x2, allpot] = poly_x2(x, n)
% expanding the feature space to include polynomial terms, 
% which allows the linear model to fit more complex patterns
    [Nx, nx] = size(x);

    % Generating all possible combinations of exponents for monomials
    if nx == 1
        % If only one regressor simply generating the powers of it
        allpot = (0:n)';
    else
        % combinations for multiple regressors
        allpot = [];
        for total_degree = 0:n
            % Generatin & appending combinations for the current total degree
            allpot = [allpot; flipud(generateCombinations(nx, total_degree))];
        end
    end
    
    nmon = size(allpot, 1); % No. of monomials is the number of rows in allpot
    
    % Preparing empty matrix for the extended regressor matrix
    x2 = ones(Nx, nmon);
    
    % Fill in the x2 matrix
    for ii = 1:nmon
        % Calculating each monomial using one row of allpot and x
        for var_idx = 1:nx
            x2(:, ii) = x2(:, ii) .* (x(:, var_idx) .^ allpot(ii, var_idx));
        end
    end
end

function combinations = generateCombinations(num_vars, total_degree)
%  a recursive helper function for poly_x2 to generate all combinations of exponents 
% for a given total degree across multiple variables.
    % all combinations of exponents for num_vars summing to total_degree
    if num_vars == 1
        combinations = total_degree;
    else
        combinations = [];
        for i = 0:total_degree
            % Recursively generating combinations for one less variable
            sub_combinations = generateCombinations(num_vars - 1, total_degree - i);
            % Pre-apending the current exponent to each combination
            combinations = [combinations; [i * ones(size(sub_combinations, 1), 1), sub_combinations]];
        end
    end
end