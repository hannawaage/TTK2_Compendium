%% The Iris Task 2 - Histogram
clear all

x1 = load('Iris_TTT4275/class_1','-ascii');
x2 = load('Iris_TTT4275/class_2','-ascii');
x3 = load('Iris_TTT4275/class_3','-ascii');

close all
figure
for i = 1:4
    subplot(4,1,i)
    hold on
    histogram(x1(:,i))
    histogram(x2(:,i))
    histogram(x3(:,i))
    hold off
    grid
    title("Feature " + i)
    legend("class 1", "class 2", "class 3")
end

%% Plotting feature 1 and 2 against each other
figure
plot(x1(:,1),x1(:,2), 'o')
hold on
plot(x2(:,1),x2(:,2), 'x')
plot(x3(:,1),x3(:,2), '+')
grid
legend('class 1', 'class 2', 'class 3')

%% Plotting feature 3 and 4 against each other
figure
plot(x1(:,3),x1(:,4), 'o')
hold on
plot(x2(:,3),x2(:,4), 'x')
plot(x3(:,3),x3(:,4), '+')
grid
legend('class 1', 'class 2', 'class 3')