% Project 1 SSY230 Learning dynamical systems using systems identification
% by Alfred Aronsson
close all; clear; clc


addpath("Given_functions\")
addpath("My_functions\")

%% Task 1 (a)


% test 1
x1 = (1:10)';
y1 = (1:10)';
test1 = LinRegress(x1,y1);

% expected theta = 1
test1.theta;
% expected variance = 0
test1.variance;

figure;
plotModel(x1,y1,test1)

x2 = [ones(length(x1),1), x1];
y2 = y1+1;

test2 = LinRegress(x2,y2);

% expected theta = 1
test2.theta;
% expected variance = 0
test2.variance;

figure;
plotModel(x2,y2,test2)

% Variance validation 1 dim
n = 10:100;
vars = zeros(length(n),100);

for j=1:100

    for i=1:length(n)

        size_1 = n(i);
        y_var = randn(size_1,1);
        x_var = ones(size_1,1);
        lr_var = LinRegress(x_var,y_var);
        vars(i,j) = lr_var.variance;
    end
end

figure;
loglog(n,mean(vars,2),'LineWidth',2)
title('Variance for 1 dim regressor')
xlabel('Sample size')
ylabel('Variance')

% Variance validation 2 dim
n = 10:100;
vars_tot = zeros(2, length(n)); 

for i = 1:length(n)
    vars = zeros(2, 100); 
    for j = 1:100 
        size_2 = n(i);
        y_var = randn(size_2, 1); 
        x_var = ones(size_2, 2); 
        x_var(2:2:end, 2) = 0; 
        x_var(:, 2) = x_var(:, 2) * 2;
        lr_var = LinRegress(x_var, y_var); 
        vars(:, j) = [lr_var.variance(1, 1); lr_var.variance(2, 2)]; 
    end
    vars_tot(:, i) = mean(vars, 2); 
end

figure;
plot(n, vars_tot(1, :), 'LineWidth', 2);
hold on; 
plot(n, 2./n, 'LineWidth', 2)
plot(n, vars_tot(2, :), 'LineWidth', 2);
plot(n, 1./n, 'LineWidth', 2)
hold off
title('Variance for 2 dim regressor'); 
xlabel('Sample size');
ylabel('Variance');
legend('Variance 1 Monte Carlo', 'Variance 1 Analytical', 'Variance 2 Monte Carlo', 'Variance 2 Analytical')

%% Task 1 (b)

y1_pred = evalModel(test1,x1);

e = y1 - y1_pred;

%% Task 1 (c)


lamda1 = 10000;
x3 = (1:10)';
y3 = (1:10)';
test3 = linRegressRegul(x3,y3,lamda1);

test3.theta
test3.variance

figure;
plotModel(x3,y3, test3);

lamda2 = 0;
test4 = linRegressRegul(x3,y3,lamda2);

figure;
plotModel(x3,y3,test4);

%% Task 1 (d)


f_quad = @(x) 1 + x + x.^2;
x4 = (1:10)';
y4 = f_quad(x4);
% Expected theta = [1; 1; 1]
test5 = polyfit(x4,y4,0,2);
test5.theta;



f_cube = @(x) 1 + x + x.^2 + x.^3;
x5 = x4;
y5 = f_cube(x5);
% Expected theta = [1; 1; 1; 1]
test6 = polyfit(x5,y5,0,3);
test6.theta;

%% Task 1 (e)

run Test_plotModel.m



%% Task 2 (c)

% Estimation data
x6 = (-5:0.5:5)';

% linear function plus noise
y6 = 3+1.5*sin(x6)+real((-1).^x6);

knn = knnRegress(x6,y6,3);
knn2 = knnRegress(x6,y6,5);
lr = LinRegress(x6,y6);

figure;
plotModel(x6,y6,knn,knn2,lr)
legend('Points','KNN_3','KNN_5','Linear Regression')

%% Task 3.1 (a)

N = [10, 100, 1000, 10000];
error_lr = zeros(length(N),1);
figure;
for i = 1:length(N)
    
    % Training data
    [x_train, y_train] = linearData(N(i),1);
    [x_train_sorted, sortedIndices_train] = sort(x_train);
    y_train_sorted = y_train(sortedIndices_train);
    X_train = [ones(length(x_train_sorted),1), x_train_sorted];

    % Noise free validation data
    [x_val, y_val] = linearData(N(i),0);
    [x_val_sorted, sortedIndices_val] = sort(x_val);
    y_val_sorted = y_val(sortedIndices_val);
    X_val = [ones(length(x_val_sorted),1), x_val_sorted];

    % True model
    true_model = LinRegress(X_val,y_val_sorted);

    % Evaluation
    lr = LinRegress(X_train, y_train_sorted);
    subplot(2,2,i)
    plotModel(X_train,y_train_sorted,lr,true_model)
    title(append('Number of data points: ',int2str(N(i))))
    legend('Training data','Regression','True Model', 'Location', 'Best')
    lr.theta;
    

end

% mse
error_lr = zeros(length(N),1);
mse_tot_50 = zeros(length(N),100);

% for i = 1:100
%     for j = 1:length(N)
%     
%     % Training data
%     [x_train, y_train] = linearData(N(j),1);
%     [x_train_sorted, sortedIndices_train] = sort(x_train);
%     y_train_sorted = y_train(sortedIndices_train);
%     X_train = [ones(length(x_train_sorted),1), x_train_sorted];
% 
%     % Noise free validation data
%     [x_val, y_val] = linearData(N(j),0);
%     [x_val_sorted, sortedIndices_val] = sort(x_val);
%     y_val_sorted = y_val(sortedIndices_val);
%     X_val = [ones(length(x_val_sorted),1), x_val_sorted];
%     lr = LinRegress(X_train, y_train_sorted);
%     error_lr(j) = mse(y_val_sorted, evalModel(lr,X_val));
% 
%     end
%     mse_tot_50(:,i) = error_lr;
% 
% end
% 
% mse_mean = mean(mse_tot_50,2);
% 
% figure;
% loglog(N,mse_mean,'LineWidth',2)
% title('MSE error for different amount of data points')
% ylabel('MSE error')
% xlabel('Number of data points')


%% Task 3.1 (b)

N = [10, 100, 1000, 10000];
figure;
for i = 1:length(N)
    
    % Training data
    [x_train, y_train] = linearData(N(i),1);
    [x_train_sorted, sortedIndices_train] = sort(x_train);
    y_train_sorted = y_train(sortedIndices_train);

    % Noise free validation data
    [x_val, y_val] = linearData(N(i),0);
    [x_val_sorted, sortedIndices_val] = sort(x_val);
    y_val_sorted = y_val(sortedIndices_val);
    X_val = [ones(length(x_val_sorted),1), x_val_sorted];

    % True model
    true_model = LinRegress(X_val,y_val_sorted);

    % Evaluation
    poly = polyfit(x_train_sorted, y_train_sorted,0,5);
    subplot(2,2,i)
    plotModel(x_train_sorted,y_train_sorted,poly,true_model)
    title(append('Number of data points: ',int2str(N(i))))
    legend('Training data','Regression','True model', 'Location', 'Best')
    poly.theta;


end

% % mse calc
% error_poly = zeros(length(N),1);
% msepoly_tot = zeros(length(N),100);
% for i = 1:100
%     for j = 1:length(N)
%     
%     % Training data
%     [x_train, y_train] = linearData(N(j),1);
%     [x_train_sorted, sortedIndices_train] = sort(x_train);
%     y_train_sorted = y_train(sortedIndices_train);
% 
%     % Noise free validation data
%     [x_val, y_val] = linearData(N(j),0);
%     [x_val_sorted, sortedIndices_val] = sort(x_val);
%     y_val_sorted = y_val(sortedIndices_val);
%     poly = polyfit(x_train_sorted, y_train_sorted,0,5);
%     error_poly(j) = mse(y_val_sorted, evalModel(poly,x_val_sorted));
% 
%     end
%     msepoly_tot(:,i) = error_poly;
% 
% end
% 
% mse_meanpoly = mean(msepoly_tot,2);
% 
% 
% figure;
% loglog(N,error_poly,'LineWidth',2)
% title('MSE error for different amount of data points')
% ylabel('MSE error')
% xlabel('Number of data points')
% 
% figure;
% loglog(N,mse_mean,'LineWidth',2)
% hold on
% loglog(N,mse_meanpoly,'LineWidth',2)
% legend('MSE Linear', 'MSE Polynomial', 'Location', 'Best')
% title('MSE error for different amount of data points')
% ylabel('MSE error')
% xlabel('Number of data points')



%% Task 3.1 (c)

N = 100; % Number of data points in each dataset
num_simulations = 100; % Number of Monte Carlo simulations
noise_stddev = 1; % Standard deviation of the noise
true_parameters = [2; -3]; % True parameters [intercept; slope]

theta_estimates = zeros(num_simulations, length(true_parameters)); % Storage for parameter estimates
var_estimates = zeros(2, num_simulations);

% Monte Carlo Simulation for parameter estimation
for i = 1:num_simulations
    % Generate dataset with noise
    X = [ones(N, 1), (1:N)']; % Design matrix [intercept; linear term]
    e = noise_stddev * randn(N, 1); % Generate random Gaussian noise
    y = X * true_parameters + e; % Responses with noise
    
    % Estimate parameters (assuming LinRegress returns a structure with a field 'theta')
    estimated_model = LinRegress(X, y);
    
    % Store the estimated parameters
    theta_estimates(i, :) = estimated_model.theta';
    var_estimates(:,i) = [estimated_model.variance(1); estimated_model.variance(4)];
end

% Empirical variances of the parameter estimates
empirical_variance_intercept = var(theta_estimates(:, 1));
empirical_variance_slope = var(theta_estimates(:, 2));

% Model variances of the parameter estimates
mean_var = mean(var_estimates,2)
model_variance_intercept = mean_var(1);
model_variance_slope = mean_var(2);

% Plot histograms of the estimated parameters
figure;
subplot(1, 2, 1); % Subplot for intercept estimates
histogram(theta_estimates(:, 1), 'BinMethod', 'auto');
title('Histogram of Intercept Estimates');
xlabel('Intercept Estimate');
ylabel('Frequency');

subplot(1, 2, 2); % Subplot for slope estimates
histogram(theta_estimates(:, 2), 'BinMethod', 'auto');
title('Histogram of Slope Estimates');
xlabel('Slope Estimate');
ylabel('Frequency');

% Display empirical variances
disp(['Empirical Variance of Intercept Estimates: ', num2str(empirical_variance_intercept)]);
disp(['Empirical Variance of Slope Estimates: ', num2str(empirical_variance_slope)]);
disp(['Mode Variance of Intercept Estimates: ', num2str(model_variance_intercept)]);
disp(['Model Variance of Slope Estimates: ', num2str(model_variance_slope)]);

%% Task 3.1 (d)

N = [10, 100, 1000];

figure;
for i = 1:length(N)
    
    % Training data
    [x_train, y_train] = linearData(N(i),1);
    [x_train_sorted, sortedIndices_train] = sort(x_train);
    y_train_sorted = y_train(sortedIndices_train);

    % Noise free validation data
    [x_val, y_val] = linearData(N(i),0);
    [x_val_sorted, sortedIndices_val] = sort(x_val);
    y_val_sorted = y_val(sortedIndices_val);

    % Evaluation
    error_lr = zeros(length(N(i)),1);
    for j = 1:N(i)
        knn = knnRegress(x_train_sorted, y_train_sorted,j);
        error_lr(j) = mse(y_val_sorted, evalModel(knn,x_val_sorted));
    end
    subplot(3,1,i)
    loglog(1:N(i),error_lr,'LineWidth',2)
    title(append(int2str(N(i)),' Data points'))
    ylabel('MSE error')
    xlabel('Neighbours')
    [~, k] = min(error_lr);

end

N = 100;
lambda = [0.01, 0.1, 1];
figure;
for i = 1:length(lambda)
    
    % Training data
    [x_train, y_train] = linearData(N,lambda(i));
    [x_train_sorted, sortedIndices_train] = sort(x_train);
    y_train_sorted = y_train(sortedIndices_train);

    % Noise free validation data
    [x_val, y_val] = linearData(N,0);
    [x_val_sorted, sortedIndices_val] = sort(x_val);
    y_val_sorted = y_val(sortedIndices_val);

    % Evaluation
    error_lr = zeros(length(N),1);
    for j = 1:N
        knn = knnRegress(x_train_sorted, y_train_sorted,j);
        error_lr(j) = mse(y_val_sorted, evalModel(knn,x_val_sorted));
    end
    subplot(3,1,i)
    loglog(1:N,error_lr,'LineWidth',2)
    title(append('Noise variance = ',num2str(lambda(i))))
    ylabel('MSE error')
    xlabel('Neighbours')
    [~, k] = min(error_lr);

end


%% Task 3.2 (a)

N = [10, 100, 1000, 10000];
error_lr = zeros(length(N),1);
figure;
for i = 1:length(N)
    
    % Training data
    [x_train, y_train] = polyData(N(i),1);
    [x_train_sorted, sortedIndices_train] = sort(x_train);
    y_train_sorted = y_train(sortedIndices_train);
    X_train = [ones(length(x_train_sorted),1), x_train_sorted];

    % Noise free validation data
    [x_val, y_val] = polyData(N(i),0);
    [x_val_sorted, sortedIndices_val] = sort(x_val);
    y_val_sorted = y_val(sortedIndices_val);
    X_val = [ones(length(x_val_sorted),1), x_val_sorted];

    % True model
    true_model = LinRegress(X_val,y_val_sorted);

    % Evaluation
    lr = LinRegress(X_train, y_train_sorted);
    subplot(2,2,i)
    plotModel(X_train,y_train_sorted,lr)
    hold on
    
    title(append('Number of data points: ',int2str(N(i))))
    legend('Validation data','Regression','','','Uncertainty', 'Location', 'best')
    xlabel('x');
    ylabel('y');
    lr.theta
    error_lr(i) = mse(y_val_sorted, evalModel(lr,X_val));

end

figure;
loglog(N,error_lr,'LineWidth',2)
title('MSE for different amount of data points')
ylabel('MSE')
xlabel('Number of data points')

%% Task 3.2 (b)


N = 15;
dim = 15;
lambda = [0, 1, 10, 100];

figure;
sgtitle('15 degree polynomial regression with different \lambda')
for i = 1:length(lambda)

    % Training data
    [x_train, y_train] = polyData(N,1);
    [x_train_sorted, sortedIndices_train] = sort(x_train);
    y_train_sorted = y_train(sortedIndices_train);


    % Validation data
    [x_val, y_val] = polyData(N,0);
    [x_val_sorted, sortedIndices_val] = sort(x_val);
    y_val_sorted = y_val(sortedIndices_val);


    % True model
    true_model = polyfit(x_val_sorted,y_val_sorted,0,5);

    % Plot
    poly = polyfit(x_train_sorted, y_train_sorted,lambda(i),dim);
    subplot(2,2,i)
    plotModel(x_train_sorted,y_train_sorted,poly,true_model)
    legend('Training data','Regression','True model', 'Location', 'best')
    title(append('\lambda = ',num2str(lambda(i))))
    poly.theta;

end


%% Task 3.2 (c)

N = [10, 100, 1000, 10000];
figure;
for i = 1:length(N)
    
    % Training data
    [x_train, y_train] = polyData(N(i),1,1);
    [x_train_sorted, sortedIndices_train] = sort(x_train);
    y_train_sorted = y_train(sortedIndices_train);
    X_train = [ones(length(x_train_sorted),1), x_train_sorted];

    % plot
    lr = LinRegress(X_train, y_train_sorted);
    subplot(2,2,i)
    plotModel(X_train,y_train_sorted,lr)
    title(append('Number of data points: ',int2str(N(i))))
    legend('Training data','Regression','','','Uncertainty', 'Location', 'best')
    xlabel('x');
    ylabel('y');
    lr.theta

end

%% Task 3.2 (d)

N = [10, 100, 1000];

figure;
for i = 1:length(N)
    
    % Training data
    [x_train, y_train] = polyData(N(i),1);
    [x_train_sorted, sortedIndices_train] = sort(x_train);
    y_train_sorted = y_train(sortedIndices_train);

    % Noise free validation data
    [x_val, y_val] = polyData(N(i),0);
    [x_val_sorted, sortedIndices_val] = sort(x_val);
    y_val_sorted = y_val(sortedIndices_val);

    % Evaluation
    error_lr = zeros(length(N(i)),1);
    for j = 1:N(i)
        knn = knnRegress(x_train_sorted, y_train_sorted,j);
        error_lr(j) = mse(y_val_sorted, evalModel(knn,x_val_sorted));
    end
    subplot(3,1,i)
    loglog(1:N(i),error_lr,'LineWidth',2)
    title(append(int2str(N(i)),' Data points'))
    ylabel('MSE error')
    xlabel('Neighbours')
    [~, k] = min(error_lr)

end

N = 100;
lambda = [0.01, 0.1, 1];
figure;
for i = 1:length(lambda)
    
    % Training data
    [x_train, y_train] = polyData(N,lambda(i));
    [x_train_sorted, sortedIndices_train] = sort(x_train);
    y_train_sorted = y_train(sortedIndices_train);

    % Noise free validation data
    [x_val, y_val] = polyData(N,0);
    [x_val_sorted, sortedIndices_val] = sort(x_val);
    y_val_sorted = y_val(sortedIndices_val);

    % Evaluation
    error_lr = zeros(length(N),1);
    for j = 1:N
        knn = knnRegress(x_train_sorted, y_train_sorted,j);
        error_lr(j) = mse(y_val_sorted, evalModel(knn,x_val_sorted));
    end
    subplot(3,1,i)
    loglog(1:N,error_lr,'LineWidth',2)
    title(append('Noise variance = ',num2str(lambda(i))))
    ylabel('MSE error')
    xlabel('Neighbours')
    [~, k] = min(error_lr)

end

%% 3.3 (a)
close all

N = 50;
dim = (2:10);
error_poly = zeros(length(dim),1);
mse_tot_50 = zeros(length(dim),100);

% Noise free validation data
[x_val, y_val] = chirpData(N,0);
[x_val_sorted, sortedIndices_val] = sort(x_val);
y_val_sorted = y_val(sortedIndices_val);

for j = 1:100

    for i = 1:length(dim)

        % Training data
        [x_train, y_train] = chirpData(N,0.4);
        [x_train_sorted, sortedIndices_train] = sort(x_train);
        y_train_sorted = y_train(sortedIndices_train);

        % Evaluation
        poly = polyfit(x_train_sorted, y_train_sorted,0,dim(i));
        error_poly(i) = mse(y_val_sorted, evalModel(poly,x_val_sorted));

    end

    mse_tot_50(:,j) = error_poly;
end

mse_mean_50 = mean(mse_tot_50,2);

[~, best_dim_50] = min(mse_mean_50);
best_dim_50 = best_dim_50+1

figure;
loglog(dim,mse_mean_50,'LineWidth',2)
title('MSE for different dim polynomial regressions')
ylabel('MSE')
xlabel('dim')

N = 1000;
dim = (2:10);
error_poly = zeros(length(dim),1);
mse_tot_1000 = zeros(length(dim),100);

% Noise free validation data
[x_val, y_val] = chirpData(N,0);
[x_val_sorted, sortedIndices_val] = sort(x_val);
y_val_sorted = y_val(sortedIndices_val);

for j = 1:100

    for i = 1:length(dim)

        % Training data
        [x_train, y_train] = chirpData(N,0.2);
        [x_train_sorted, sortedIndices_train] = sort(x_train);
        y_train_sorted = y_train(sortedIndices_train);

        % Evaluation
        poly = polyfit(x_train_sorted, y_train_sorted,0,dim(i));
        error_poly(i) = mse(y_val_sorted, evalModel(poly,x_val_sorted));

    end

    mse_tot_1000(:,j) = error_poly;
end

mse_mean_1000 = mean(mse_tot_1000,2);

[~, best_dim_1000] = min(mse_mean_1000);
best_dim_1000 = best_dim_1000+1

figure;
loglog(dim,mse_mean_1000,'LineWidth',2)
title('MSE for different dim polynomial regressions')
ylabel('MSE')
xlabel('dim')

%% 3.3 (b)
close all

figure;
loglog(dim,error_poly,'LineWidth',2)
title('MSE for different dim polynomial regressions')
ylabel('MSE')
xlabel('dim')

N = 100;
dim = 2:10;
lambda = [0, 0.01, 0.1, 10];
error_poly = zeros(length(dim), length(lambda));

for l = 1:length(lambda)



    figure;
    sgtitle(append('\lambda = ',num2str(lambda(l))))
    for i = 1:length(dim)

        % Training data
        [x_train, y_train] = chirpData(N,1);
        [x_train_sorted, sortedIndices_train] = sort(x_train);
        y_train_sorted = y_train(sortedIndices_train);

        % Noise free validation data
        [x_val, y_val] = chirpData(N,0);
        [x_val_sorted, sortedIndices_val] = sort(x_val);
        y_val_sorted = y_val(sortedIndices_val);

        % Evaluation
        poly = polyfit(x_train_sorted, y_train_sorted,lambda(l),dim(i));
        subplot(3,3,i)
        plotModel(x_val_sorted,y_val_sorted,poly)
        legend('Validation data','Regression', 'Location', 'best')
        title(append('dim: ',int2str(dim(i))))
        poly.theta;
        error_poly(i,l) = mse(y_val_sorted, evalModel(poly,x_val_sorted));

    end


end

figure;
for l = 1:length(lambda)
    
    loglog(dim,error_poly(:,l),'LineWidth',2,'DisplayName',append('\lambda: ', num2str(lambda(l))))
    hold on

end

title('MSE for different dim and regularization polynomial regressions')
ylabel('MSE')
xlabel('dim')
legend('Location','east')

%% Task 3.3 (c)

N = 100;


% Training data
[x_train, y_train] = chirpData(N,1);
[x_train_sorted, sortedIndices_train] = sort(x_train);
y_train_sorted = y_train(sortedIndices_train);

% Noise free validation data
[x_val, y_val] = chirpData(N,0);
[x_val_sorted, sortedIndices_val] = sort(x_val);
y_val_sorted = y_val(sortedIndices_val);

% Evaluation
error_lr = zeros(length(N),1);
for j = 1:N
    knn = knnRegress(x_train_sorted, y_train_sorted,j);
    error_lr(j) = mse(y_val_sorted, evalModel(knn,x_val_sorted));
end
[~, k] = min(error_lr)

poly = polyfit(x_train_sorted, y_train_sorted,0.1,10);

figure;
loglog(1:N,error_lr,'LineWidth',2)
ylabel('MSE error')
xlabel('Neighbours')

MSE_poly = mse(y_val_sorted, evalModel(poly,x_val_sorted))
MSE_knn = mse(y_val_sorted, evalModel(knnRegress(x_train_sorted,y_train_sorted,k),x_val_sorted))

%% Task 4.1 (a)

N = 100;

% Training data
[x_vis, y_vis] = twoDimData1(N,0);


[xq,yq] = meshgrid(0:0.2:10, 0:0.2:10);

vq = griddata(x_vis(:,1),x_vis(:,2),y_vis,xq,yq);
figure;
mesh(xq,yq,vq);

hold on
plot3(x_vis(:,1),x_vis(:,2),y_vis,'o');

%% Task 4.1 (b)

N_train = 100;
N_validate = 1000;

[x_train, y_train] = twoDimData1(N_train,1);
[x_val, y_val] = twoDimData1(N_validate,0);

figure;
scatter3(x_train(:,1),x_train(:,2),y_train)
title('Training data')

figure;
scatter3(x_val(:,1),x_val(:,2),y_val)
title('Validation data')

% Task 4.1 (c)



lr = LinRegress(x_train,y_train);

lr.theta;
lr.variance;

quality_lr = mse(y_val,evalModel(lr,x_val))

%Task 4.1 (d)

poly2 = polyfit(x_train,y_train,0,2);
quality2 = mse(y_val,evalModel(poly2,x_val)) % poly2 best model

poly3 = polyfit(x_train,y_train,0,3);
quality3 = mse(y_val,evalModel(poly3,x_val))

poly4 = polyfit(x_train,y_train,0,4);
quality4 = mse(y_val,evalModel(poly4,x_val))

poly5 = polyfit(x_train,y_train,0,5);
quality5 = mse(y_val,evalModel(poly5,x_val))



% Task 4.1 (e)

error_lr = zeros(length(N_train),1);
for j = 1:N_train
    knn = knnRegress(x_train, y_train,j);
    error_lr(j) = mse(y_val, evalModel(knn,x_val));
end
[~, k] = min(error_lr)

figure;
loglog(1:N_train,error_lr,'LineWidth',2)
title(append(int2str(N_train), ' Training data'));
ylabel('MSE error')
xlabel('Neighbours')

MSE_poly = quality2
MSE_knn = mse(y_val, evalModel(knnRegress(x_train,y_train,k),x_val))



