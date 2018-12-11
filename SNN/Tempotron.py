# coding=utf-8
import numpy as np
import pickle
import matplotlib.pyplot as plt
import bisect
import time
import collections
import scipy.io as scio
import random
from sklearn import metrics


class Tempotron(object):
    def __init__(self, tau, tau_s, n_in, n_class, n_neuron_per_class, **kwargs):
        seed = kwargs.get('seed', 123)
        weight_variance = kwargs.get('weight_variance', 0.1)
        weight_offset = kwargs.get('weight_offset', 0)
        step_T = kwargs.get('step_T', None)
        end_T = kwargs.get('end_T', 1.0)
        lmd = kwargs.get('lmd', 2e-2)  # lambda
        threshold = kwargs.get('threshold', 1.0)
        min_event = kwargs.get('min_event', 10)

        # np.random.seed(seed)
        self._n_in = n_in  # 输入的神经元通道数
        self._n_class = n_class
        self._n_neuron_per_class = n_neuron_per_class  # 每一类所使用的神经元个数
        self._n_neuron = n_class * n_neuron_per_class
        self._Weights = weight_variance * np.random.randn(n_in, n_class, n_neuron_per_class) + weight_offset

        self._end_T = end_T
        self._step_T = step_T if step_T else end_T * 0.001
        self._tau = tau
        self._tau_s = tau_s if tau_s else tau / 5
        self._T = np.arange(0, end_T + self._step_T, self._step_T)
        self._v_lut1 = np.exp(-self._T / tau)
        self._v_lut2 = np.exp(-self._T / tau_s)
        self._V = self._v_lut1 - self._v_lut2
        self._v_coe = 1. / np.max(self._v_lut1 - self._v_lut2)  # 对数差的最大值
        self._argmax_v_t = np.argmax(self._v_lut1 - self._v_lut2) * self._step_T  # 对数差最大值的时间点

        self._lmd = lmd
        self._threshold = threshold * np.ones((self._n_class, self._n_neuron_per_class))
        self._min_event = min_event

    def _lut1(self, t):
        return np.exp(-t / self._tau)

    def _lut2(self, t):
        return np.exp(-t / self._tau_s)

    def _fire(self, V_max):
        return V_max >= self._threshold

    def K(self, t):  # kernel函数，最大值为1
        return (self._lut1(t) - self._lut2(t)) * self._v_coe

    def _vote(self, fire):  # 统计每个类别中发放的神经元总数，并返回最大值的类别（下标）
        counts = np.sum(fire, axis=1)
        return [idx for idx, count in enumerate(counts) if count == np.max(counts)]

    def _oneHot(self, label):  # 对应标签的行值为1
        tgt = np.zeros((self._n_class, self._n_neuron_per_class), dtype=np.int32)
        tgt[label] = np.ones((1, self._n_neuron_per_class), dtype=np.int32)
        return tgt

    def _init_data(self, addr, time):  # addr——对应通道  在做什么？
        time = [t - time[0] for t in time]  # 将time变成从0开始的时间间隔差
        end_idx = bisect.bisect_left(time, self._end_T)  # 截取end_T时间长度的发放时间段
        addr, time = addr[:end_idx], time[:end_idx]
        idx = bisect.bisect_left(time, self._end_T - self._argmax_v_t)
        addr = list(addr) + [-1] * idx  # 用列表拼接起来
        time = list(time) + [t + self._argmax_v_t for t in time[:idx]]
        time, addr = zip(*sorted(zip(time, addr), key=lambda x: x[0]))  # 按照元组中第一个元素值进行排序
        time, addr = np.array(time), np.array(addr, dtype=np.int32)
        return addr, time

    def _find_last_event(self, addr):
        return np.where(addr >= 0)[0][-1]  # 最后一次发放的下标数

    def _find_fire_event(self, v):
        fire_event = -1 * np.ones((self._n_class, self._n_neuron_per_class), dtype=np.int32)
        v_max = np.max(v, axis=0)  # 连续电压的最大值 v:[发放脉冲数，类别，神经元]-3维
        for i in range(self._n_class):
            for j in range(self._n_neuron_per_class):
                if v_max[i, j] >= self._threshold[i, j]:
                    fire_event[i, j] = np.where(v[:, i, j] >= self._threshold[i, j])[0][0]  # 返回的是第一个超过阈值的脉冲下标
        return fire_event

    def _simulate_fires(self, addr, time):
        v = self.simulate(addr, time)
        v_max = np.max(v, axis=0)
        return self._fire(v_max)

    def simulate(self, addr, time):
        assert isinstance(addr, np.ndarray)
        number_of_spike = addr.shape[0]
        v_lut1 = np.zeros((number_of_spike, self._n_class, self._n_neuron_per_class), dtype=np.float32)
        v_lut2 = np.zeros((number_of_spike, self._n_class, self._n_neuron_per_class), dtype=np.float32)
        v_lut1[0] = v_lut2[0] = self._Weights[addr[0]]
        for i in range(1, number_of_spike):
            delta_t = time[i] - time[i - 1]
            delta_v = self._Weights[addr[i]] if addr[i] >= 0 else 0
            v_lut1[i] = v_lut1[i - 1] * self._lut1(delta_t) + delta_v  # 这个公式的来源是什么？
            v_lut2[i] = v_lut2[i - 1] * self._lut2(delta_t) + delta_v
        return (v_lut1 - v_lut2) * self._v_coe

    def feed_forward(self, addr, time):
        v = self.simulate(addr, time)
        v_max = np.max(v, axis=0)
        v_idx = np.where(v_max >= self._threshold, self._find_fire_event(v), self._find_last_event(addr))
        return v_max, v_idx

    def _update(self, addr, time, label):
        if len(addr) < self._min_event: return
        addr, time = self._init_data(addr, time)
        v_max, v_idx = self.feed_forward(addr, time)
        t_max = time[v_idx]
        delta = -1 * self._fire(v_max)  # delta代表参数变化的正负方向
        delta[label] += 1
        for k in range(np.max(v_idx)):
            if addr[k] >= 0:
                self._Weights[addr[k], :, :] += self._lmd * self.K(np.maximum(t_max - time[k], 0)) * delta

    def train(self, addrs, times, labels):
        number_of_samples = len(addrs)
        tr_order = random.sample(range(number_of_samples), number_of_samples)
        for i in tr_order:
            self._update(addrs[i], times[i], labels[i])

    def test(self, addrs, times, labels, all=False):
        number_of_samples = len(addrs)
        ACC, T, TP, FP = list(), list(), list(), list()
        PREDICT = list()
        for i in range(number_of_samples):
            addr, time, label = addrs[i], times[i], labels[i]
            addr, time = self._init_data(addr, time)
            fire = self._simulate_fires(addr, time)
            is_fire = (fire > 0)
            tp_cnt = np.sum(is_fire[label])
            fp_cnt = np.sum(is_fire) - tp_cnt
            t_cnt = (self._n_class - 1) * self._n_neuron_per_class - fp_cnt + tp_cnt
            voted = self._vote(fire)
            predict = random.choice(voted or range(self._n_class))  # 这个random.choice的意义在哪里
            acc = (voted and random.choice(voted) == label)

            ACC.append(acc)
            T.append(t_cnt * 1. / self._n_neuron)
            TP.append(tp_cnt * 1. / self._n_neuron_per_class)
            FP.append(fp_cnt * 1. / (self._n_neuron - self._n_neuron_per_class))
            PREDICT.append(predict)
        return (np.mean(ACC), np.mean(T), np.mean(TP), np.mean(FP)) if not all else \
            (np.mean(ACC), np.mean(T), np.mean(TP), np.mean(FP), PREDICT)

    def loss(self):
        pass

    def save(self, filepath):
        assert isinstance(filepath, str)
        with open(filepath, 'wb') as f:
            pickle.dump({'Weights': self._Weights, 'n_in': self._n_in, 'n_class': self._n_class,
                         'n_neuron_per_class': self._n_neuron_per_class, 'threshold': self._threshold}, f)

    def load(self, filepath):
        assert isinstance(filepath, str)
        try:
            with open(filepath, 'rb') as f:
                d = pickle.load(f)
                self._Weights = d['Weights']
                self._n_in = d['n_in']
                self._n_class = d['n_class']
                self._n_neuron_per_class = d['n_neuron_per_class']
                self._threshold = d['threshold']
                self._n_neuron = self._n_class * self._n_neuron_per_class
        except:
            raise AssertionError('Unsatisfied filepath {}'.format(filepath))


class AdaptiveTempotron(Tempotron):
    def __init__(self, tau, tau_s, n_in, n_class, n_neuron_per_class, **kwargs):
        # kwargs['threshold'] = 4.0
        super(AdaptiveTempotron, self).__init__(tau=tau, tau_s=tau_s, n_in=n_in, n_class=n_class,
                                                n_neuron_per_class=n_neuron_per_class, **kwargs)

    def _func(self, v):
        return np.log(np.exp(v) + 1)

    def _func_dirivative(self, v):
        return 1 / (np.exp(-v) + 1)  # 为什么是-v

    def feed_forward(self, addr, time):
        v = self.simulate(addr, time)
        v_max = np.max(v, axis=0)
        v_idx = np.argmax(v, axis=0)
        return v_max, v_idx  # 直接就返回最大的电压值及下标

    def _update(self, addr, time, label):
        if len(addr) < self._min_event: return
        addr, time = self._init_data(addr, time)
        v_max, v_idx = self.feed_forward(addr, time)
        t_max = time[v_idx]
        func_v = self._func(v_max)
        dirivative_v = self._func_dirivative(v_max)
        func_v_sum = np.sum(func_v, axis=0, keepdims=True)

        delta = (-1. / func_v_sum) * np.ones((self._n_class, self._n_neuron_per_class), dtype=np.float32)  # 对应数值相乘
        delta[label] += 1. / func_v[label]
        delta *= dirivative_v

        for k in range(np.max(v_idx)):
            if addr[k] >= 0:
                self._Weights[addr[k], :, :] += self._lmd * self.K(np.maximum(t_max - time[k], 0)) * delta  # 直接就更新吗

    def _simulate_fires(self, addr, time):
        fires = np.zeros((self._n_class, self._n_neuron_per_class), dtype=np.int32)
        number_of_spike = addr.shape[0]
        v_lut1 = v_lut2 = self._Weights[addr[0]] * self._v_coe
        for i in range(1, number_of_spike):
            delta_t = time[i] - time[i - 1]
            delta_v = self._Weights[addr[i]] * self._v_coe if addr[i] >= 0 else 0
            v_lut1 = v_lut1 * self._lut1(delta_t) + delta_v
            v_lut2 = v_lut2 * self._lut2(delta_t) + delta_v
            fire = (v_lut1 - v_lut2) > self._threshold  # 这里没乘v_coe
            fires = fires + fire
            v_lut1 = v_lut1 - fire * self._threshold
            if np.any(v_lut1) < 0: print(v_lut1)
        return fires

        # fires = np.zeros((self._n_class, self._n_neuron_per_class), dtype=np.int32)
        # number_of_spikes = len(addr)
        # for i in range(self._n_class):
        #     for j in range(self._n_neuron_per_class):
        #         v_lut1 = v_lut2 = self._Weights[addr[0], i, j] * self._v_coe
        #         for k in range(1, number_of_spikes):
        #             delta_t = time[k] - time[k - 1]
        #             delta_v = self._Weights[addr[k], i, j]*self._v_coe if addr[k] >= 0 else 0
        #             v_lut1 = v_lut1 * self._lut1(delta_t) + delta_v
        #             v_lut2 = v_lut2 * self._lut2(delta_t) + delta_v
        #             if v_lut1 - v_lut2 > self._threshold[i,j]:
        #                 fires[i, j] += 1
        #                 if v_lut1 < 0:
        #                     print(v_lut1, v_lut2)
        #                 v_lut1 -= self._threshold[i,j]
        # return fires


def odor_tempotron_random(odor, rand_num, iter_num, timescale, train_sample_num):
    type = 'AdaptiveTempotron'
    #type = 'Tempotron'
    labelSeq = range(1, len(odor)+1)
    label = odor
    numClass = len(odor)
    totalSample = 20 * numClass
    trial = range(1, 21)
    channel = range(1, 12)
    #train_sample_num = 16
    test_sample_num = 20 - train_sample_num
    n_neuron_per_class = 10
    ACC, PREDICT, TRUE = list(), list(), list()
    TRAIN_ACC, TRAIN_PREDICT, TRAIN_TRUE = list(), list(), list()
    addr, time, y = list(), list(), list()

    for l in labelSeq:
        for t in trial:
            y.append(l - 1)
            cur_addr, cur_time = list(), list()
            for c in channel:
                fname = 'snn_data/timescale_{}/spike-time-{}-{}-{}-{}.mat'.format(float(timescale), label[l - 1], t, c, float(timescale))
                try:
                    f = scio.loadmat(fname)
                    for ts in f['spikeTime'][0]:
                        cur_addr.append(c - 1)
                        cur_time.append(ts)
                except:
                    print(fname)
            cur_time, cur_addr = zip(*sorted(zip(cur_time, cur_addr), key=lambda x: x[0]))
            time.append(list(cur_time))  # 不同类别所有样本的全部通道发放时间序列
            addr.append(list(cur_addr))  # 发放时间点所对应的通道名
    for i in range(len(time)):
        for j in range(len(time[i])):
            time[i][j] -= 8  # 将时间值缩小

    idx_grouped_by_labels = [[], [], [], []]
    for i, l in enumerate(y):
        idx_grouped_by_labels[l].append(i)

    for randNum in range(rand_num):
        test_idx_grouped_by_labels = [random.sample(idx_grouped_by_labels[i], 20) for i in range(numClass)]
        tempotron = eval(
            "{}(5, 1, 11, numClass, n_neuron_per_class, end_T=5, step_T=0.001, lmd={},seed=1, threshold=1)".format(
                type, 0.001 if type == 'AdaptiveTempotron' else 0.01))
        cur_test_idx = [test_idx_grouped_by_labels[j][k] for j in range(numClass) for k in range(0, test_sample_num)]
        tr_addr, tr_time, tr_y = [addr[j] for j in range(totalSample) if j not in cur_test_idx], \
                                 [time[j] for j in range(totalSample) if j not in cur_test_idx], \
                                 [y[j] for j in range(totalSample) if j not in cur_test_idx]
        te_addr, te_time, te_y = [addr[j] for j in range(totalSample) if j in cur_test_idx], \
                                 [time[j] for j in range(totalSample) if j in cur_test_idx], \
                                 [y[j] for j in range(totalSample) if j in cur_test_idx]
        for iterNum in range(iter_num):
            tempotron.train(tr_addr, tr_time, tr_y)
            tr_acc_temp, T, TP, FP, predicted = tempotron.test(tr_addr, tr_time, tr_y, all=True)
            print(odor),
            print '%s, rand_num %d, temp_iter %d, train_acc %.3f' % (type, randNum, iterNum, tr_acc_temp)

        tr_acc, T, TP, FP, predicted = tempotron.test(tr_addr, tr_time, tr_y, all=True)
        te_acc, T, TP, FP, predicted = tempotron.test(te_addr, te_time, te_y, all=True)
        print 'train_acc & test_acc of cur rand_num %d' % randNum
        print '%s, rand_num %d, train_acc %.3f' % (type, randNum, tr_acc)
        print '%s, rand_num %d, test_acc %.3f' % (type, randNum, te_acc)
        ACC.append(te_acc)
        # PREDICT.extend(max_predicted)
        # TRUE.extend(te_y)
        TRAIN_ACC.append(tr_acc)
        # TRAIN_PREDICT.extend(max_train_predicted)
        # TRAIN_TRUE.extend(tr_y)

    print 'train', TRAIN_ACC
    # print metrics.confusion_matrix(TRAIN_TRUE, TRAIN_PREDICT)
    print 'test', ACC
    # print metrics.confusion_matrix(TRUE, PREDICT)
    print

    return TRAIN_ACC, ACC


def odor_tempotron_curve(odor, iter_num, timescale):
    type = 'AdaptiveTempotron'
    #type = 'Tempotron'
    labelSeq = range(1, len(odor)+1)
    label = odor
    numClass = len(odor)
    totalSample = 20 * numClass
    trial = range(1, 21)
    channel = range(1, 12)
    train_sample_num = 16
    test_sample_num = 20 - train_sample_num
    n_neuron_per_class = 10
    ACC, PREDICT, TRUE = list(), list(), list()
    TRAIN_ACC, TRAIN_PREDICT, TRAIN_TRUE = list(), list(), list()
    addr, time, y = list(), list(), list()

    for l in labelSeq:
        for t in trial:
            y.append(l - 1)
            cur_addr, cur_time = list(), list()
            for c in channel:
                fname = 'snn_data/timescale_{}/spike-time-{}-{}-{}-{}.mat'.format(float(timescale), label[l - 1], t, c, float(timescale))
                try:
                    f = scio.loadmat(fname)
                    for ts in f['spikeTime'][0]:
                        cur_addr.append(c - 1)
                        cur_time.append(ts)
                except:
                    print(fname)
            cur_time, cur_addr = zip(*sorted(zip(cur_time, cur_addr), key=lambda x: x[0]))
            time.append(list(cur_time))  # 不同类别所有样本的全部通道发放时间序列
            addr.append(list(cur_addr))  # 发放时间点所对应的通道名
    for i in range(len(time)):
        for j in range(len(time[i])):
            time[i][j] -= 8  # 将时间值缩小

    idx_grouped_by_labels = [[], [], [], []]
    for i, l in enumerate(y):
        idx_grouped_by_labels[l].append(i)


    test_idx_grouped_by_labels = [random.sample(idx_grouped_by_labels[i], 20) for i in range(numClass)]
    tempotron = eval(
        "{}(5, 1, 11, numClass, n_neuron_per_class, end_T=5, step_T=0.001, lmd={},seed=1, threshold=1)".format(
            type, 0.001 if type == 'AdaptiveTempotron' else 0.01))
    cur_test_idx = [test_idx_grouped_by_labels[j][k] for j in range(numClass) for k in range(0, test_sample_num)]
    tr_addr, tr_time, tr_y = [addr[j] for j in range(totalSample) if j not in cur_test_idx], \
                             [time[j] for j in range(totalSample) if j not in cur_test_idx], \
                             [y[j] for j in range(totalSample) if j not in cur_test_idx]
    te_addr, te_time, te_y = [addr[j] for j in range(totalSample) if j in cur_test_idx], \
                             [time[j] for j in range(totalSample) if j in cur_test_idx], \
                             [y[j] for j in range(totalSample) if j in cur_test_idx]
    for iterNum in range(iter_num):
        tempotron.train(tr_addr, tr_time, tr_y)
        tr_acc, T, TP, FP, predicted = tempotron.test(tr_addr, tr_time, tr_y, all=True)
        te_acc, T, TP, FP, predicted = tempotron.test(te_addr, te_time, te_y, all=True)
        print(odor),
        print '%s, temp_iter %d, train_acc %.3f' % (type, iterNum, tr_acc)
        print '%s, temp_iter %d, test_acc %.3f' % (type, iterNum, te_acc)
        ACC.append(te_acc)
        # PREDICT.extend(max_predicted)
        # TRUE.extend(te_y)
        TRAIN_ACC.append(tr_acc)
        # TRAIN_PREDICT.extend(max_train_predicted)
        # TRAIN_TRUE.extend(tr_y)

    print 'train', TRAIN_ACC
    # print metrics.confusion_matrix(TRAIN_TRUE, TRAIN_PREDICT)
    print 'test', ACC
    # print metrics.confusion_matrix(TRUE, PREDICT)
    print

    return TRAIN_ACC, ACC