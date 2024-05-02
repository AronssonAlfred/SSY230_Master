function y_sim = idsimulate(model, u)
    % Convert ARX model to transfer function
    transfer = id2tf(model);
    

    t = (0:length(u)-1)';  % Create a time vector assuming unit time steps    

    num_states = max(model.na, model.nb-1 + model.nk);

    x0 = zeros(num_states, 1);

    % Simulate the model response
    y_sim = lsim(ss(transfer), u, t,x0);
end