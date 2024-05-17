%% Sketch for project 3 in SSY230 by Alfred Aronsson
close all; clear; clc

% Load files with input data and validation data
ESTIM = load('F16Data_FullMSine_Level3.mat'); 
VALI = load('F16Data_FullMSine_Level2_Validation.mat');

%% Consider SISO-case with force as input and acceleration at node 1 as output

% Estimation data
output_ESTIM = ESTIM.Acceleration(1,:)';
input_ESTIM = ESTIM.Force';

% Validation data
output_VALI = VALI.Acceleration(1,:)';
input_VALI = VALI.Force';

systemIdentification