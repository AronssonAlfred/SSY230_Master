%% Sketch for project 2 in SSY230, by Alfred Aronsson
close all; clear; clc;

addpath("My_functions\");

%% Task 1 (a)

% end-ny-rows+1:end-ny

% Validation test on student_test_pr2.m
run student_test_pr2.m

% Validation test using filter
input_sequence = ones(20, 1); 

b = 0.5; 
a = [1 -0.5]; 

output_sequence = filter(b, a, input_sequence);

na = 1; 
nb = 1; 
nk = 0; 

data_matrix = [output_sequence input_sequence];

estimated_model = arxfit(data_matrix, [na, nb, nk]);

disp('Known filter coefficients:');
disp(['a: ', num2str(a)]);
disp(['b: ', num2str(b)]);

disp('Estimated ARX model coefficients:');
disp(['A: ', num2str(estimated_model.theta(1:estimated_model.na+1)')]);
disp(['B: ', num2str(estimated_model.theta(estimated_model.na+2:end)')]);

%% Task 1 (b)

u = [1 2 3]';
y = [1 2 3]';

model=arxfit([y u],[1 1 0]);

ypred = idpredict(model, [y u], 1)

%% Task 1 (c)


