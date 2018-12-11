function [train_accu, test_accu] = knnClassifyRandom(dataSet, dataLabel, class)
    fold = 5;
    sampleSize = 20;
    sampleOneFold = sampleSize / fold;
    dataNum = (1:size(dataLabel,1))';
    train_accu = 0;
    test_accu = 0;
    iter = 100;
%     KNNModel = fitcknn(dataSet, dataLabel, 'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('optimizer','bayesopt','AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',MaxObjectiveEvaluations,'kfold',kfold,'ShowPlots',0));
%     CVKNNModel = crossval(KNNModel, 'kfold', kfold);
%     classLoss = kfoldLoss(CVKNNModel);
%     accuracy = 1 - classLoss;
    for i = 1:1:iter
        testSeq = zeros(class*sampleOneFold,0);
        for j = 1:1:class
            testSeqTemp = randperm(sampleSize, sampleOneFold)';
            testSeqTemp = testSeqTemp + (j-1)*sampleSize;
            testSeq = [testSeq; testSeqTemp];            
        end
        testSeq = sort(testSeq);
        trainSeq = setdiff(dataNum, testSeq);
        train_data = dataSet(trainSeq, :);
        train_label = dataLabel(trainSeq, :);
        test_data = dataSet(testSeq, :);
        test_label = dataLabel(testSeq, :);
        KNN_model = fitcknn(train_data, train_label, 'NumNeighbors', 3, 'Standardize', 1);
        
        predict_label = predict(KNN_model, train_data);
        num_train_sample = size(train_data, 1);
        count = 0;
        for j = 1:1:num_train_sample
            if predict_label(j) == train_label(j)
                count = count + 1;
            end
        end
        train_accu = train_accu + count / num_train_sample;
        
        predict_label = predict(KNN_model, test_data);
        num_test_sample = size(test_data, 1);
        count = 0;
        for j = 1:1:num_test_sample
            if predict_label(j) == test_label(j)
                count = count + 1;
            end
        end
        test_accu = test_accu + count / num_test_sample;
    end
    
    train_accu = train_accu / iter;
    test_accu = test_accu / iter;
end