function [x,y] = linearData(N,var,stoch)
%[x,y] =linearData(N,var) This function generates data for project 1. You
% indicate the number of data with N, and var indicates the variance of the
% additive Gaussian white noise.
%
% You can give a third argument stoch, if it is set to 0 you obtain
% deterministic equal distributed x instead of stochastic equally
% distributed. 
%
% [x,y] = linearData(N,var,stoch)
%

if nargin<3, stoch=1; end

if stoch
  x=rand(N,1)*10;
else
    x=(10*(1:N)/N)';
end

y=1.5+0.5*x+sqrt(var)*randn(N,1);
end

