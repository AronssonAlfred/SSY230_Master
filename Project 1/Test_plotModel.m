% This m-files  tests your functions 
% plotModel, LinRegress, polyfit, knnRegress
%
% it generates three plots and they should be identical to the plots in
% the appendix of  project 1 instructions, if everything is correct.

close all; clear; clc

rng(1)

addpath("Given_functions\")
addpath("My_functions\")
% Estimation data
x=(-5:0.5:5)';

% linear function plus noise
y=3+1.5*sin(x)+real((-1).^x);

m=LinRegress(x,y);
poly = polyfit(x,y,0,3);
poly2 = polyfit(x,y,0,10);
poly2.theta
knn = knnRegress(x,y,3);


x2=(-5:0.1:5)';
figure
plotModel(x,y,m)

figure
plotModel(x,y,m,poly,knn)

figure
plotModel(x,y,poly)
