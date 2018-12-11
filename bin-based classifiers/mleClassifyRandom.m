function [train_accu, test_accu] = mleClassify(dataSet, dataLabel, class)
    
    [total_sample_num, total_feature_num] = size(dataSet);
    iter = 100;
    channel_num = 1;
    one_sample_num = 20;
    fold = 5;
    test_sample_num = one_sample_num / fold;
    train_sample_num = one_sample_num - test_sample_num;
    feature_per_channel = total_feature_num / channel_num;
    dataSetTemp = dataSet;
    dataSet = zeros(total_sample_num, channel_num);
    for i = 1:1:total_sample_num
        for j = 1:1:channel_num
            dataSet(i,j) = mean(dataSetTemp(i,(j-1)*feature_per_channel+1:j*feature_per_channel));
        end
    end
    
    sample_order = [1:1:total_sample_num]';
    train_accu = 0;
    test_accu = 0;
    for r = 1:1:iter
        test_order = zeros(class*test_sample_num, 0);
        for i = 1:1:class
            test_order_temp = randperm(one_sample_num, test_sample_num)';
            test_order_temp = test_order_temp + (i-1) * one_sample_num;
            test_order_temp = sort(test_order_temp);
            test_order = [test_order; test_order_temp];
        end
        train_order = setdiff(sample_order, test_order);
        test_data = dataSet(test_order,:);
        test_label = dataLabel(test_order,:);
        train_data = dataSet(train_order,:);
        train_label = dataLabel(train_order,:);

        mle_model = zeros(class, channel_num, 3);
        for i = 1:1:class
            for j = 1:1:channel_num
                train_data_temp = train_data((i-1)*train_sample_num+1:i*train_sample_num, j);
                mle_model(i, j, 1) = mean(train_data_temp);
                mle_model(i, j, 2) = std(train_data_temp);
                mle_model(i, j, 3) = train_label(i*train_sample_num);
            end
        end
        
        pro_model = ones(class*train_sample_num, class, 2);
        count = 0;
        for i = 1:1:class*train_sample_num
            for j = 1:1:class
                for k = 1:1:channel_num
                    pro_model(i,j,1) = pro_model(i,j,1) * normpdf(train_data(i,k), mle_model(j,k,1), mle_model(j,k,2));
                    pro_model(i,j,2) = mle_model(j,k,3);
                end
            end
            [M,I] = max(pro_model(i,:,1));
            if(pro_model(i,I,2) == train_label(i))
                count = count + 1;
            end
        end
        train_accu = train_accu + count / (class*train_sample_num);
        
        pro_model = ones(class*test_sample_num, class, 2);
        count = 0;
        for i = 1:1:class*test_sample_num
            for j = 1:1:class
                for k = 1:1:channel_num
                    pro_model(i,j,1) = pro_model(i,j,1) * normpdf(test_data(i,k), mle_model(j,k,1), mle_model(j,k,2));
                    pro_model(i,j,2) = mle_model(j,k,3);
                end
            end
            [M,I] = max(pro_model(i,:,1));
            if(pro_model(i,I,2) == test_label(i))
                count = count + 1;
            end
        end
        test_accu = test_accu + count / (class*test_sample_num);
    end
    
    train_accu = train_accu / iter;
    test_accu = test_accu / iter;
end
