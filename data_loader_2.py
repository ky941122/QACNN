#coding=utf-8
from __future__ import division

import numpy as np

def batch_iter(data, batch_size, num_epochs, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            #start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            start_index = end_index-batch_size
            yield shuffled_data[start_index:end_index]



def read_data(filename, seq_len, pad_id):
    pad_id = int(pad_id)
    data = []
    f = open(filename, "r")
    for line in f.readlines():
        line = line.strip()
        q, a1, a2 = line.split("\t")

        q = q.strip()
        q = q.split()
        q = q[:seq_len]
        q = q + [pad_id] * (seq_len - len(q))

        a1 = a1.strip()
        a1 = a1.split()
        a1 = a1[:seq_len]
        a1 = a1 + [pad_id] * (seq_len - len(a1))

        a2 = a2.strip()
        a2 = a2.split()
        a2 = a2[:seq_len]
        a2 = a2 + [pad_id] * (seq_len - len(a2))

        data.append((q, a1, a2))

    print "read data done..."
    return data



def read_data2(filename, seq_len, pad_id):
    pad_id = int(pad_id)
    data = []
    f = open(filename, "r")
    s = 0
    for line in f.readlines():
        line = line.strip()
        q, a1, a2 = line.split("\t")

        q = q.strip()
        q, q_tag = q.split("###")
        q = q.strip()
        q_tag = q_tag.strip()

        q = q.split()
        lq = len(q)
        q = q[:seq_len]
        q = q + [pad_id] * (seq_len - len(q))

        q_tag = q_tag.split()
        lqt = len(q_tag)

        if lq != lqt:
            s += 1
            continue

        q_tag = q_tag[:seq_len]
        q_tag = q_tag + [pad_id] * (seq_len - len(q_tag))



        a1 = a1.strip()
        a1, a1_tag = a1.split("###")
        a1 = a1.strip()
        a1_tag = a1_tag.strip()

        a1 = a1.split()
        la1 = len(a1)
        a1 = a1[:seq_len]
        a1 = a1 + [pad_id] * (seq_len - len(a1))

        a1_tag = a1_tag.split()
        la1t = len(a1_tag)

        if la1 != la1t:
            s += 1
            continue

        a1_tag = a1_tag[:seq_len]
        a1_tag = a1_tag + [pad_id] * (seq_len - len(a1_tag))



        a2 = a2.strip()
        a2, a2_tag = a2.split("###")
        a2 = a2.strip()
        a2_tag = a2_tag.strip()

        a2 = a2.split()
        la2 = len(a2)
        a2 = a2[:seq_len]
        a2 = a2 + [pad_id] * (seq_len - len(a2))

        a2_tag = a2_tag.split()
        la2t = len(a2_tag)

        if la2 != la2t:
            s += 1
            continue

        a2_tag = a2_tag[:seq_len]
        a2_tag = a2_tag + [pad_id] * (seq_len - len(a2_tag))

        data.append((q, a1, a2, q_tag, a1_tag, a2_tag))

    print "read data done..., Skip", s, "lines!!!"
    return data



