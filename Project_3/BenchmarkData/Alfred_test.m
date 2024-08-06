%% Sketch for project 3 in SSY230 by Alfred Aronsson
close all; clear; clc

% Load files with input data and validation data
ESTIM1 = load('F16Data_FullMSine_Level3.mat'); 
VALI1 = load('F16Data_FullMSine_Level2_Validation.mat');

ESTIM2 = load('F16Data_FullMSine_Level5.mat'); 
VALI2 = load('F16Data_FullMSine_Level4_Validation.mat');

ESTIM3 = load('F16Data_FullMSine_Level7.mat'); 
VALI3 = load('F16Data_FullMSine_Level6_Validation.mat');

%% Consider SISO-case with force as input and acceleration at mode 1 as output

% Estimation data
output_ESTIM1 = ESTIM1.Acceleration(1,:)';
input_ESTIM1 = ESTIM1.Voltage';

output_ESTIM2 = ESTIM2.Acceleration(1,:)';
input_ESTIM2 = ESTIM2.Voltage';

output_ESTIM3 = ESTIM3.Acceleration(1,:)';
input_ESTIM3 = ESTIM3.Voltage';

% Validation data
output_VALI1 = VALI1.Acceleration(1,:)';
input_VALI1 = VALI1.Voltage';

output_VALI2 = VALI2.Acceleration(1,:)';
input_VALI2 = VALI2.Voltage';

output_VALI3 = VALI3.Acceleration(1,:)';
input_VALI3 = VALI3.Voltage';

systemIdentification