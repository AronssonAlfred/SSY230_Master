function idcompare(model, z, horizon)



    t = (1:length(z))';

    y_pred = idpredict(model, z, horizon);
    y_sim = idsimulate(model, z(:,2));

    figure;
    % Prediction plot
    subplot(2,1,1)
    plot(t,z(:,1))
    hold on
    plotModel(t,y_pred,model)
    title('Prediction')
    legend('Meassured', 'Predicted','','', 'Uncertainty')
    hold off

    % Simulation plot
    subplot(2,1,2)
    plot(t,z(:,1))
    hold on
    plotModel(t,y_sim,model)
    title('Simulation')
    legend('Meassured', 'Simulated', '','','Uncertainty')







end