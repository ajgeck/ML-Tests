%% Using SVM on a Two-Dimensional Multivariate Gaussian
% Parameters for the normal distributions
mu1    = [3; 6];
sigma1 = [1/2 0; 0 2];
mu2    = [3; -2];
sigma2 = [2 0; 0 2];

% Samples
samples = 1000;

% Set the seed for reproducibility
rng(436);

% Generate random data points for each class
data1 = mvnrnd(mu1, sigma1, samples);
data2 = mvnrnd(mu2, sigma2, samples);

% Combine the data from both classes into one dataset
X = [data1; data2];
% Labels for the classes
Y = [ones(samples, 1); -ones(samples, 1)];

% Dual Problem Variables
H = (Y*Y') .* (X*X');
f = -ones(2*samples, 1);
A = -eye(2*samples);
b = zeros(2*samples, 1);
Aeq = Y';
beq = 0;

% Bounds for the λ variables (0 <= λi <= C)
C = 1; % Set C = 1 for simplicity
lb = zeros(2*samples, 1);
ub = C * ones(2*samples, 1);

% Use quadprog to compute lambda
lambda = quadprog(H, f, A, b, Aeq, beq, lb, ub);

% Identify support vectors
support_vectors = find(lambda > 0.00001);

% Calculate w
w = X' * (lambda .* Y);

% Calculate w0 using support vectors
s  = support_vectors(1); % First support vector
w0 = 1/Y(s) - w' * X(s,:)';

% Plot the data points
scatter(data1(:, 1), data1(:, 2), 'r'); 
hold on;
scatter(data2(:, 1), data2(:, 2), 'b');
scatter(X(support_vectors, 1), X(support_vectors, 2), 'g', 'filled');

% Plot the decision boundary
x1_plot = min(X(:, 1)):0.1:max(X(:, 1));
x2_plot = (-w(1) * x1_plot - w0)/w(2);
plot(x1_plot, x2_plot, 'k', 'LineWidth', 2);

% Add labels and title for clarity
xlabel('X1');
ylabel('X2');
title('SVM Classifier with Support Vectors and Decision Boundary');
legend({'Class 1', 'Class 2', 'Support Vectors', 'Decision Boundary'}, 'Location', 'best');

hold off;
