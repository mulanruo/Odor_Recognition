clear all; clc;

mean_test = zeros(10, 3);
count = 0;
for timescale = 0.1:0.1:1.0
    count = count + 1;
    file_name = sprintf('odor_2&3&4-classification_MLE_random_timescale_%.1f.mat', timescale);
    load(file_name);
    mean_test(count, 1) = mean(mean(accu(2, :, 1:6)));
    mean_test(count, 2) = mean(mean(accu(2, :, 7:10)));
    mean_test(count, 3) = mean(mean(accu(2, :, 11)));
end