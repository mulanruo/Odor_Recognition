function [dataSetTemp, dataLabel] = dataGenerate_channel_4(gas, bin, timeLength, dimension)

    % 系统参数设置：bin\timeLength\dimension\gas
    %bin = 0.6;
    %timeLength = 4.8;
    %dimension = 10;
    %gas = [1,2];

    %程序主体
    timescale = bin * 40000;
    if timeLength > 0
        binNum = ceil(timeLength/bin);
    else
        binNum = floor(timeLength/bin);
    end
    channel = 6;
    sampleSize = 20;
    channelNum = length(channel);
    lengthGas = length(gas);
    dataSet = zeros(0,0); dataLabel = zeros(0,0);

    for i = 1:1:lengthGas
        for sample = 1:1:sampleSize
            datasetTemp = zeros(1, 0);
            for j = 1:1:channelNum
                V = zeros(1,abs(binNum));
                filename = sprintf('spike-%d-%d-%d.mat',gas(i),sample,channel(j));
                load(filename);
                filename = sprintf('time-%d-%d.mat',gas(i),sample);
                load(filename);
                startTime = KBD1(1) * 40000;
                endTime = startTime + binNum * timescale;
                if endTime < startTime
                    tempTime = endTime;
                    endTime = startTime;
                    startTime = tempTime;
                end
                for temp = 1:1:abs(binNum)
                    V(temp) =  length(find(spike>=(startTime+(temp-1)*timescale) & spike <= (startTime+temp*timescale))); 
                end
                datasetTemp = [datasetTemp,V];
            end
            dataSet = [dataSet;datasetTemp];
        end
        dataLabel = [dataLabel;ones(sampleSize,1)*gas(i)];
    end
    dataSetTemp = dataSet;
    
    if dimension < sampleSize
%         dataSetTemp = zeros(0,0);
%         for i = 1:1:lengthGas
%             %temp = zscore(dataSet(((i-1)*sampleSize+1):i*sampleSize,:));
%             temp = dataSet(((i-1)*sampleSize+1):i*sampleSize,:);
%             [coef,score] = pca(temp);
%             temp = score(:,1:dimension);
%             dataSetTemp = [dataSetTemp;temp];
%         end
         [coef,score] = pca(dataSet);
         dataSetTemp = score(:,1:dimension);
    end
    %dataset = zscore(dataset);
    %[coef,score] = pca(dataset);
    %dataset = score(:,1:dimension);
    %dataSet = dataset;
    %filename = sprintf('dataset-%d-%.1f-%.1f.mat', gass, bin, timeLength);
    %save(filename, 'data');
    
end