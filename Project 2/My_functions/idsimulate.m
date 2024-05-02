function y_sim = idsimulate(model, u)
    % Convert ARX model to transfer function
    transfer = id2tf(model);
    
    % Extract input data from z
    t = (1:length(u))';  % Create a time vector assuming unit time steps    

    num_states = max(model.na, model.nb-1 + model.nk);

    x0 = zeros(num_states, 1);

    % Simulate the model response
    y_sim = lsim(transfer, u, t, x0);
end