function [accuracy] = knnClassify(dataSet, dataLabel, class)
    fold = 5;
    sampleSize = 20;
    sampleOneFold = sampleSize / fold;
    dataNum = (1:size(dataLabel,1))';
    accuracy = 0;
%     KNNModel = fitcknn(dataSet, dataLabel, 'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('optimizer','bayesopt','AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',MaxObjectiveEvaluations,'kfold',kfold,'ShowPlots',0));
%     CVKNNModel = crossval(KNNModel, 'kfold', kfold);
%     classLoss = kfoldLoss(CVKNNModel);
%     accuracy = 1 - classLoss;
    for i = 1:1:fold
        testSeq = zeros(class*sampleOneFold,0);
        for j = 1:1:class
            testSeqTemp = (((i-1)*sampleOneFold+1+sampleSize*(j-1)):1:i*sampleOneFold+sampleSize*(j-1))';
            testSeq = [testSeq; testSeqTemp];            
        end
        trainSeq = setdiff(dataNum, testSeq);
        train_data = dataSet(trainSeq, :);
        train_label = dataLabel(trainSeq, :);
        test_data = dataSet(testSeq, :);
        test_label = dataLabel(testSeq, :);
        KNN_model = fitcknn(train_data, train_label, 'NumNeighbors', 3, 'Standardize', 1);
        predict_label = predict(KNN_model, test_data);
        num_test_sample = size(test_data, 1);
        count = 0;
        for j = 1:1:num_test_sample
            if predict_label(j) == test_label(j)
                count = count + 1;
            end
        end
        accuracy = accuracy + count / num_test_sample;
    end
    
    accuracy = accuracy / fold;
end