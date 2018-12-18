clear all;
clc;

%bin = 0.5;
timeLength = 5.0;
dimension = 40;

accu = zeros(2,10,11);
bincount = 0;
for bin = 0.1:0.1:1.0    
    bincount = bincount + 1;
    count = 0;
    class = 2;
    for i = 1:1:3
        for j = (i+1):1:4
            count = count + 1;
            gas = [i,j];
            [dataSet, dataLabel] = dataGenerate(gas, bin, timeLength, dimension);
            %accu(bincount, count) = knnClassify(dataSet, dataLabel, class);
            [accu(1, bincount, count), accu(2, bincount, count)] = knnClassifyRandom(dataSet, dataLabel, class);
        end
    end
end

bincount = 0;
for bin = 0.1:0.1:1.0    
    bincount = bincount + 1;
    count = 6;
    class = 3;
    for i = 1:1:2
        for j = (i+1):1:3
            for k = (j+1):1:4
                count = count + 1;
                gas = [i,j,k];
                [dataSet, dataLabel] = dataGenerate(gas, bin, timeLength, dimension);
                %accu(bincount, count) = knnClassify(dataSet, dataLabel, class);
                [accu(1, bincount, count), accu(2, bincount, count)] = knnClassifyRandom(dataSet, dataLabel, class);
            end
        end
    end
end

bincount = 0;
for bin = 0.1:0.1:1.0    
    bincount = bincount + 1;
    class = 4;
    gas = [1,2,3,4];
    [dataSet, dataLabel] = dataGenerate(gas, bin, timeLength, dimension);
    %accu(bincount) = knnClassify(dataSet, dataLabel, class);
    [accu(1, bincount, 11), accu(2, bincount, 11)] = knnClassifyRandom(dataSet, dataLabel, class);
end
save('odor_2&3&4-classification_KNN_random.mat', 'accu');