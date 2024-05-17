function m = LinRegressRegul(x, y, lambda)
%extending linregress to prevent overfitting
%by appending scaled identity matrices and zero vectors, respectively 
% This effectively adds a regularization term (λθ^2) to the least squares loss function

    % Appending extra rows to x and y for regularization
    x2 = [x; sqrt(lambda)*eye(size(x,2))];
    y2 = [y; zeros(size(x,2), 1)];
    
    % LinRegress function on the updated x2 and y2
    m = LinRegress(x2, y2);
end