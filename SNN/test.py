#!/usr/bin/env python
# coding=utf-8
import numpy as np
import random
from Tempotron import odor_tempotron_random, odor_tempotron_curve
import matplotlib.pyplot as plt

rand_num = 20
iter_num = 25
timescale = 0.9
timeinterval = 0.1
starttime = 0.1
trainset = 16

## timescale from 0.1 to 1.0 && 1.0 to 5.0
# for temp_timescale in range(1, timescale+timeinterval):
# for temp_timescale in np.arange(starttime, timescale+timeinterval, timeinterval):
#     train_acc_set = list()
#     test_acc_set = list()
#     for i in range(1, 4):
#         for j in range(i+1, 5):
#             odor = [i, j]
#             train_acc, test_acc = odor_tempotron_random(odor, rand_num, iter_num * 2, temp_timescale)
#             train_acc_set.append(train_acc)
#             test_acc_set.append(test_acc)
#
#     for i in range(1, 3):
#         for j in range(i+1, 4):
#             for k in range(j+1, 5):
#                 odor = [i, j, k]
#                 train_acc, test_acc = odor_tempotron_random(odor, rand_num, iter_num * 3, temp_timescale)
#                 train_acc_set.append(train_acc)
#                 test_acc_set.append(test_acc)
#
#     odor = [1, 2, 3, 4]
#     train_acc, test_acc = odor_tempotron_random(odor, rand_num, iter_num * 4, temp_timescale)
#     train_acc_set.append(train_acc)
#     test_acc_set.append(test_acc)
#
#     fname = 'adaptive_train_acc_timescale_{}.npy'.format(float(temp_timescale))
#     np.save(fname, train_acc_set)
#     fname = 'adaptive_test_acc_timescale_{}.npy'.format(float(temp_timescale))
#     np.save(fname, test_acc_set)

# for temp_timescale in np.arange(0.1, 0.3, 0.1):
#     fname = 'results/adaptive_train_acc_timescale_{}.npy'.format(float(temp_timescale))
#     train_acc = np.load(fname)
#     fname = 'results/adaptive_test_acc_timescale_{}.npy'.format(float(temp_timescale))
#     test_acc = np.load(fname)
#     mean_train_acc = np.mean(train_acc, axis=1)
#     mean_test_acc = np.mean(test_acc, axis=1)
#     np.set_printoptions(precision=2)
#     print mean_train_acc
#     print mean_test_acc
#     print

## trainset from 1 to 16

# for temp_trainset in range(1, trainset):
#     train_acc_set = list()
#     test_acc_set = list()
#     for i in range(1, 4):
#         for j in range(i+1, 5):
#             odor = [i, j]
#             train_acc, test_acc = odor_tempotron_random(odor, rand_num, iter_num * 2, 5, temp_trainset)
#             train_acc_set.append(train_acc)
#             test_acc_set.append(test_acc)
#
#     for i in range(1, 3):
#         for j in range(i+1, 4):
#             for k in range(j+1, 5):
#                 odor = [i, j, k]
#                 train_acc, test_acc = odor_tempotron_random(odor, rand_num, iter_num * 3, 5, temp_trainset)
#                 train_acc_set.append(train_acc)
#                 test_acc_set.append(test_acc)
#
#     odor = [1, 2, 3, 4]
#     train_acc, test_acc = odor_tempotron_random(odor, rand_num, iter_num * 4, 5, temp_trainset)
#     train_acc_set.append(train_acc)
#     test_acc_set.append(test_acc)
#
#     fname = 'adaptive_train_acc_trainset_{}.npy'.format(temp_trainset)
#     np.save(fname, train_acc_set)
#     fname = 'adaptive_test_acc_trainset_{}.npy'.format(temp_trainset)
#     np.save(fname, test_acc_set)

# for temp_trainset in np.arange(1, 16, 1):
#     fname = 'results/adaptive_train_acc_trainset_{}.npy'.format(temp_trainset)
#     train_acc = np.load(fname)
#     fname = 'results/adaptive_test_acc_trainset_{}.npy'.format(temp_trainset)
#     test_acc = np.load(fname)
#     mean_train_acc = np.mean(train_acc, axis=1)
#     mean_test_acc = np.mean(test_acc, axis=1)
#     np.set_printoptions(precision=2)
#     print mean_train_acc
#     print mean_test_acc
#     print


# different time periods

file_path = 'results/var_timescale/'

for temp_timescale in np.arange(1.0, 6.0, 1.0):
    test_acc = np.load(file_path+'adaptive_test_acc_timescale_{}.npy'.format(temp_timescale))
    test_acc = np.mean(test_acc, axis=1)
    np.set_printoptions(precision=2)
    print test_acc

for temp_timescale in np.arange(5.0, 6.0, 1.0):
    test_acc = np.load(file_path+'general_test_acc_timescale_{}.npy'.format(temp_timescale))
    test_acc = np.mean(test_acc, axis=1)
    np.set_printoptions(precision=2)
    print test_acc


# different training sets

file_path = 'results/var_trainset/'
file_path_2 = 'results/var_trainset_2/'
trainset_num = 17
twoclass_test_acc = list()
threeclass_test_acc = list()
fourclass_test_acc = list()
twoclass_test_acc_2 = list()
threeclass_test_acc_2 = list()
fourclass_test_acc_2 = list()

for temp_trainset in range(1, trainset_num):
    temp_test_acc = np.load(file_path + 'adaptive_test_acc_trainset_{}.npy'.format(temp_trainset))
    temp_test_acc = np.mean(temp_test_acc, axis=1)
    temp_twoclass_test_acc = np.mean(temp_test_acc[0:6])
    temp_threeclass_test_acc = np.mean(temp_test_acc[6:10])
    temp_fourclass_test_acc = temp_test_acc[10]
    twoclass_test_acc.append(temp_twoclass_test_acc)
    threeclass_test_acc.append(temp_threeclass_test_acc)
    fourclass_test_acc.append(temp_fourclass_test_acc)

    temp_test_acc = np.load(file_path_2 + 'adaptive_test_acc_trainset_{}.npy'.format(temp_trainset))
    temp_test_acc = np.mean(temp_test_acc, axis=1)
    temp_twoclass_test_acc = np.mean(temp_test_acc[0:6])
    temp_threeclass_test_acc = np.mean(temp_test_acc[6:10])
    temp_fourclass_test_acc = temp_test_acc[10]
    twoclass_test_acc_2.append(temp_twoclass_test_acc)
    threeclass_test_acc_2.append(temp_threeclass_test_acc)
    fourclass_test_acc_2.append(temp_fourclass_test_acc)

plt.figure()
x = range(1, trainset_num)
plt.plot(x, twoclass_test_acc)
plt.plot(x, twoclass_test_acc_2)
plt.xlabel('Size of Training Set')
plt.ylabel('Test Accuracy')
plt.title('Two-Class Average Odor Discrimination Accuracy of Different Training Sets')
plt.legend(labels = ['Tempotron-VR', 'classical Tempotron'], loc='lower right')
plt.show()

plt.figure()
x = range(1, trainset_num)
plt.plot(x, threeclass_test_acc)
plt.plot(x, threeclass_test_acc_2)
plt.xlabel('Size of Training Set')
plt.ylabel('Test Accuracy')
plt.title('Three-Class Average Odor Discrimination Accuracy of Different Training Sets')
plt.legend(labels = ['Tempotron-VR', 'classical Tempotron'], loc='lower right')
plt.show()

plt.figure()
x = range(1, trainset_num)
plt.plot(x, fourclass_test_acc)
plt.plot(x, fourclass_test_acc_2)
plt.xlabel('Size of Training Set')
plt.ylabel('Test Accuracy')
plt.title('Four-Class Odor Discrimination Accuracy of Different Training Sets')
plt.legend(labels = ['Tempotron-VR', 'classical Tempotron'], loc='lower right')
plt.show()


# training set & test set curve

iter_num = 20
train_acc_12, test_acc_12 = list(), list()
train_acc_124, test_acc_124 = list(), list()
train_acc_1234, test_acc_1234 = list(), list()
file_path = 'results/train_test_curve/'

# odor = [1, 2]
# for i in range(iter_num):
#     train_acc, test_acc = odor_tempotron_curve(odor, 100, 5)
#     train_acc_12.append(train_acc)
#     test_acc_12.append(test_acc)
# np.save('adaptive_train_acc_curve_12.npy', train_acc_12)
# np.save('adaptive_test_acc_curve_12.npy', test_acc_12)
general_train_acc_12 = np.load(file_path + 'general_train_acc_curve_12.npy')
general_test_acc_12 = np.load(file_path + 'general_test_acc_curve_12.npy')
general_train_acc = np.mean(general_train_acc_12, axis=0)
general_test_acc = np.mean(general_test_acc_12, axis=0)
adaptive_train_acc_12 = np.load(file_path + 'adaptive_train_acc_curve_12.npy')
adaptive_test_acc_12 = np.load(file_path + 'adaptive_test_acc_curve_12.npy')
adaptive_train_acc = np.mean(adaptive_train_acc_12, axis=0)
adaptive_test_acc = np.mean(adaptive_test_acc_12, axis=0)
plt.figure()
plt.plot(adaptive_train_acc)
plt.plot(general_train_acc)
plt.title("Training Set Accuracy Curve of Odor Combination 1-2 for 100 Iterations")
plt.xlim((0, 100))
plt.ylim((0.4, 1.02))
plt.xlabel('Number of Iteration')
plt.ylabel('Training Accuracy')
plt.legend(labels = ['Tempotron-VR', 'classical Tempotron'], loc='lower right')
plt.show()
plt.figure()
plt.plot(adaptive_test_acc)
plt.plot(general_test_acc)
plt.title("Test Set Accuracy Curve of Odor Combination 1-2 for 100 Iterations")
plt.xlim((0, 100))
plt.ylim((0.4, 1.02))
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy')
plt.legend(labels = ['Tempotron-VR', 'classical Tempotron'], loc='lower right')
plt.show()

# odor = [1, 2, 4]
# for i in range(iter_num):
#     train_acc, test_acc = odor_tempotron_curve(odor, 100, 5)
#     train_acc_124.append(train_acc)
#     test_acc_124.append(test_acc)
# np.save('adaptive_train_acc_curve_124.npy', train_acc_124)
# np.save('adaptive_test_acc_curve_124.npy', test_acc_124)
general_train_acc_124 = np.load(file_path + 'general_train_acc_curve_124.npy')
general_test_acc_124 = np.load(file_path + 'general_test_acc_curve_124.npy')
general_train_acc = np.mean(general_train_acc_124, axis=0)
general_test_acc = np.mean(general_test_acc_124, axis=0)
adaptive_train_acc_124 = np.load(file_path + 'adaptive_train_acc_curve_124.npy')
adaptive_test_acc_124 = np.load(file_path + 'adaptive_test_acc_curve_124.npy')
adaptive_train_acc = np.mean(adaptive_train_acc_124, axis=0)
adaptive_test_acc = np.mean(adaptive_test_acc_124, axis=0)
plt.figure()
plt.plot(adaptive_train_acc)
plt.plot(general_train_acc)
plt.title("Training Set Accuracy Curve of Odor Combination 1-2-4 for 100 Iterations")
plt.xlim((0, 100))
plt.ylim((0.3, 1.02))
plt.xlabel('Number of Iteration')
plt.ylabel('Training Accuracy')
plt.legend(labels = ['Tempotron-VR', 'classical Tempotron'], loc='lower right')
plt.show()
plt.figure()
plt.plot(adaptive_test_acc)
plt.plot(general_test_acc)
plt.title("Test Set Accuracy Curve of Odor Combination 1-2-4 for 100 Iterations")
plt.xlim((0, 100))
plt.ylim((0.3, 1.02))
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy')
plt.legend(labels = ['Tempotron-VR', 'classical Tempotron'], loc='lower right')
plt.show()

# odor = [1, 2, 3, 4]
# for i in range(iter_num):
#     train_acc, test_acc = odor_tempotron_curve(odor, 100, 5)
#     train_acc_1234.append(train_acc)
#     test_acc_1234.append(test_acc)
# np.save('adaptive_train_acc_curve_1234.npy', train_acc_1234)
# np.save('adaptive_test_acc_curve_1234.npy', test_acc_1234)
general_train_acc_1234 = np.load(file_path + 'general_train_acc_curve_1234.npy')
general_test_acc_1234 = np.load(file_path + 'general_test_acc_curve_1234.npy')
general_train_acc = np.mean(general_train_acc_1234, axis=0)
general_test_acc = np.mean(general_test_acc_1234, axis=0)
adaptive_train_acc_1234 = np.load(file_path + 'adaptive_train_acc_curve_1234.npy')
adaptive_test_acc_1234 = np.load(file_path + 'adaptive_test_acc_curve_1234.npy')
adaptive_train_acc = np.mean(adaptive_train_acc_1234, axis=0)
adaptive_test_acc = np.mean(adaptive_test_acc_1234, axis=0)
plt.figure()
plt.plot(adaptive_train_acc)
plt.plot(general_train_acc)
plt.title("Training Set Accuracy Curve of Odor Combination 1-2-3-4 for 100 Iterations")
plt.xlim((0, 100))
plt.ylim((0.2, 1.02))
plt.xlabel('Number of Iteration')
plt.ylabel('Training Accuracy')
plt.legend(labels = ['Tempotron-VR', 'classical Tempotron'], loc='lower right')
plt.show()
plt.figure()
plt.plot(adaptive_test_acc)
plt.plot(general_test_acc)
plt.title("Test Set Accuracy Curve of Odor Combination 1-2-3-4 for 100 Iterations")
plt.xlim((0, 100))
plt.ylim((0.2, 1.02))
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy')
plt.legend(labels = ['Tempotron-VR', 'classical Tempotron'], loc='lower right')
plt.show()
