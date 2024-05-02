echo on
% This file illustrates how you can illustrate the distribution of an
% estimate using Monte Carlo techniques. This requires that you can repeat
% the stochastic realization many times and investigate the statistics of
% the outcome. This is normally no problem when you computer generate your
% data with a new noise realization each time. However, in "real"
% situations where it is costly to perform experements, it is not a
% possible method.

disp('press a button')
pause

% If you can repeat your estimation and use fresh data each time, data
% having a new, independent disturbance contribution each time, then you
% can numerically estimate the distribution of your estimator by estimating
% many estimates and investigate its propertiess.

disp('press a button')
pause

% Here we illustrate the idea on a simple example, you should apply the
% idea to the parameter estimate in your models, and try to validate the
% analytic expression for the parameter uncertainty.

% Consider the case we have an estimator of a variable e and we think it is
% noramlly distributed with standard deviation 3 and mean value 2.

disp('press a button')
pause

% Let us estimate it three times. Since the estimates are based on
% different independent data, we obtain three estimates

e=3*randn(3,1)+2

disp('press a button')
pause

% We can illustrate the outcome in a histogram

histogram(e,3)

% This does not say very much, 3 samples is too little to estimate a
% distribution.

% you need to increase the number of samples

disp('press a button')
pause

% Here you have a script based on N realizations. The result is displayed
% as a histogram together with the assumed distribution function.  

% You can change N and repeat the calculations. Make it 10, 100, 1000 times
% larger, does the histogram converge to the assumed distribution?

% Note, the amplitude of the assumed distribution function is not properly
% tuned. It has bin set in just to make the plot look good.

N=600; % number of samples
bins=N/100;
e=3*randn(N,1)+2;
sigm=3; % standard devisation
histogram(e,bins)
hold on
x=(-10:0.1:10)+2;
y=2900*(1/(sigm*sqrt(2*pi)))*exp(-((x-2)/sigm).^2/2);
plot(x,y)

hold off
