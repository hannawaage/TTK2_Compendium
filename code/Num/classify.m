%% Classification of the 10K test images using all 60K 
% training images as templates. 
% Remember to load 'data_all.mat' before running.

classified = NaN(num_test, 1);

% Choose between NN and KNN 
knn = false;
K = 7;

% Number of partitions of the test data 
num_partitions = 20;

% This for loop partitions the classification into a job of 500 images 
% at once to avoid too large matrices. Also speeds up the process.
for i = 1:num_partitions
    start = 1 + (i-1)*num_test/num_partitions;
    endin = i*num_test/num_partitions;
    x = testv(start:endin, :);  
    % M becomes a matrix of all the 500 images' distance to the 60K
    % training images
    M = pdist2(x, trainv);
    k = 1;
    % Find the template(s) with the smallest distance and classify 
    % the test image as the same as this
    if knn
       for j = start:endin
            [~, I] = mink(M(k, :), K);
            for n = 1:length(I)
                I(n) = trainlab(I(n));
            end
            right_class = mode(I);
            classified(j) = right_class;
            k = k + 1;
        end 
    else
        for j = start:endin
            [~, I] = min(M(k, :));
            classified(j) = trainlab(I);  
            k = k + 1;
        end
    end
end

% Calculate confusion matrix and the error rate 
% for the classifier on the test set
C = confusionmat(classified, testlab); 

error_rate = (num_test-trace(C))/num_test;



