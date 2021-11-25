%% The Iris Task 1
clear all

%% Setup
x1 = load('Iris_TTT4275/class_1','-ascii');
x2 = load('Iris_TTT4275/class_2','-ascii');
x3 = load('Iris_TTT4275/class_3','-ascii');

Ntest   = 30;           % Number of samples for testing
i       = [3,4];        % Features to include
maxiter = 1000;         % max iterations in gradient descent method
alpha   = 0.01;         % Fixed step length in gradient descent method

% Use first 30 samples to train:
xtrain = [x1(1:Ntest,i); x2(1:Ntest,i); x3(1:Ntest,i)];
xtest  = [x1(Ntest+1:end,i); x2(Ntest+1:end,i); x3(Ntest+1:end,i)];

% Use last 30 samples to train:
% xtrain = [x1(end-Ntest+1:end,i);x2(end-Ntest+1:end,i);x3(end-Ntest+1:end,i)];
% xtest  = [x1(1:end-Ntest,i); x2(1:end-Ntest,i); x3(1:end-Ntest,i)];

C            = 3;                                        % Number of classes
D            = size(xtrain,2);                           % Dimension of x
W0           = 0.01*ones(C,D+1);                         % Initial weights
sigmoid      = @(x) 1./(1+exp(-x));                      
discriminant = @(xk,W) sigmoid(W*xk);                    
X            = [xtrain(1:Ntest*C,:)'; ones(1, Ntest*C)]; % Feature matrix
T            = [kron(ones(1,Ntest), [1 0 0]') ...        % Target values
                kron(ones(1,Ntest), [0 1 0]')  ...
                kron(ones(1,Ntest), [0 0 1]')];             

%% Training
W = gradient_descent(@(W)MSE_grad(X,T,W,discriminant), W0, alpha, maxiter);

%% Testing
Xtest       = [xtest'; ones(1, size(xtest,1))];               % Test cases
Ttest       = [repelem(1,20), repelem(2,20), repelem(3,20)];  % Target
[~,classes] = max(W*Xtest);                                   % Prediction

error_rate  = sum(classes~=Ttest)/length(classes)
confusion   = confusionmat(Ttest,classes)

% Finding error rate and confusion matrix for training data
Ttrain              = [repelem(1,30), repelem(2,30), repelem(3,30)];
[~,trained_classes] = max(W*X);
error_rate_train    = sum(trained_classes~=Ttrain)/length(trained_classes)
confusion_train     = confusionmat(Ttrain, trained_classes)


