%% Classification of the 10K test images using generated 
% clusters as templates. 
% Remember to load 'data_all.mat' before running.

% Number of templates in each cluster
M = 64;

% Choose between NN and KNN 
knn = false;
K = 7;

% Number of partitions of the test data 
num_partitions = 20;

%% Find indeces of the vectors of the individual classes
class_length = ones(1, 10);
max_num = 6743;
class_indeces = NaN(10, max_num);
for i = 1:num_train
    cl = trainlab(i) + 1;
    class_indeces(cl, class_length(1, cl)) = i;
    class_length(1, cl) = class_length(1, cl) + 1;
end

%% Find class specific clusters - Cis is a 3D Matrix of
% cluster matrices, each w/ 64 templates 
Cis = NaN(M, 10, vec_size);
for j = 1:10
    n = class_length(j) - 1;
    classVectors = NaN(n, vec_size);
    k = 1; 
    while(~isnan(class_indeces(j, k)))
        ind = class_indeces(j, k);
        classVectors(k, :) = trainv(ind, :);
        k = k + 1;
    end
    [~, Cis(:, j, :)] = kmeans(classVectors, M);
end
C = reshape(Cis, [10*M vec_size]);
%% Find smallest distance 
classified = NaN(num_test, 1);
for i = 1:num_partitions
    start = 1 + (i-1)*num_test/num_partitions;
    endin = i*num_test/num_partitions;
    x = testv(start:endin, :);
    M_dist = pdist2(x, C);
    k = 1;
    if knn
        for j = start:endin
            [~, I] = mink(M_dist(k, :), K);
            for l = 1:length(I)
                I(l) = floor((I(l)-1)/M);
            end
            right_class = mode(I);
            classified(j) = right_class;
            k = k + 1;
        end
    else
        for j = start:endin
            [~, I] = min(M_dist(k, :));
            I = floor((I-1)/M);
            classified(j) = I;
            k = k + 1;
        end
    end
    
end

% Calculate confusion matrix and the error rate 
% for the classifier on the test set
Conf = confusionmat(classified, testlab); 

error_rate = (num_test-trace(Conf))/num_test;