clear all; clc;

load('odor_2&3&4-classification_DA_random_timescale_1.0.mat');
train_acc_1 = reshape(accu(1,:,:), 10, 11);
test_acc_1 = reshape(accu(2,:,:), 10, 11);

load('odor_2&3&4-classification_DA_random_timescale_2.0.mat');
train_acc_2 = reshape(accu(1,:,:), 10, 11);
test_acc_2 = reshape(accu(2,:,:), 10, 11);

load('odor_2&3&4-classification_DA_random_timescale_3.0.mat');
train_acc_3 = reshape(accu(1,:,:), 10, 11);
test_acc_3 = reshape(accu(2,:,:), 10, 11);

load('odor_2&3&4-classification_DA_random_timescale_4.0.mat');
train_acc_4 = reshape(accu(1,:,:), 10, 11);
test_acc_4 = reshape(accu(2,:,:), 10, 11);

load('odor_2&3&4-classification_DA_random_timescale_5.0.mat');
train_acc_5 = reshape(accu(1,:,:), 10, 11);
test_acc_5 = reshape(accu(2,:,:), 10, 11);

mean_test_acc_1 = mean(test_acc_1);
plot(mean_test_acc_1);
hold on;
mean_test_acc_2 = mean(test_acc_2);
plot(mean_test_acc_2);
hold on;
mean_test_acc_3 = mean(test_acc_3);
plot(mean_test_acc_3);
hold on;
mean_test_acc_4 = mean(test_acc_4);
plot(mean_test_acc_4);
hold on;
mean_test_acc_5 = mean(test_acc_5);
plot(mean_test_acc_5);
hold on;
legend('1','2','3','4','5');
