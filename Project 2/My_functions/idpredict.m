function y_pred = idpredict(model, z, horizon)

    Y = z(:,1);
    U = z(:,2);
% 
%     u=[1 -1 2 -1 3 0 1]';
%     y=[2 -2 -3 -4 -2 0 1]';


    len = length(Y);

    A = model.theta(1:model.na)';
    B = model.theta(model.na+1:end)';

    
    y_sum = zeros(len,1);
    u_sum = zeros(len,1);


    for i = 1:model.na
        x = zeros(len,1); % x is just a intermediate variable
        y = A(i).*Y(1:end-i);
        x(i+1:length(Y)) = y;
        y_sum = y_sum + x;
    end


    for i = 1:model.nb      
        x = zeros(len,1);
        u = B(i).*U(1:end-i);
        x(i+1:length(U)) = u;
        u_sum = u_sum + x;
    end


    y_pred = u_sum + y_sum; % u_sum - y_sum also gives the wrong result
    
end

