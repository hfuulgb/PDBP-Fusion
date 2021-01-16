# -*- coding: utf-8 -*-
import numpy as np
import pdb


def read_seq(seq_file):
    seq_list = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            seq = line[:-1]
            # seq_array = get_seq_concolutional_array(seq)
            seq_list.append(seq)

    return np.array(seq_list)

def read_seq_onehot(seq_file):
    seq_list = []
    seq = ''
    with open(seq_file, 'r') as fp:
        index = 0
        for line in fp:
            index += 1
            seq = line[:-1]
            seq_array = get_seq_concolutional_array(seq)
            seq_list.append(seq_array)
        print(index)
    return np.array(seq_list)

def get_seq_concolutional_array_v1(seq):
    seq = seq.replace('B', 'Z')
    seq = seq.replace('J', 'Z')
    seq = seq.replace('O', 'Z')
    seq = seq.replace('U', 'Z')
    seq = seq.replace('X', 'Z')
    alpha = 'ACDEFGHIKLMNPQRSTVWYZ'
    row = (len(seq))
    new_array = np.zeros((row, 20))
    for i, val in enumerate(seq):
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except ValueError:
            pdb.set_trace()
    return new_array

def get_seq_concolutional_array(seq):
    # seq = seq.replace('U', 'T')
    alpha = 'ACDEFGHIKLMNPQRSTVWY'
    row = (len(seq))
    new_array = np.zeros((row, 20))

    for i, val in enumerate(seq):

        if val not in 'ACDEFGHIKLMNPQRSTVWY':
            if val == 'Z':
                new_array[i] = np.array([0.0] * 20)
            # if val == 'S':
            #     new_array[i] = np.array([0, 0.5, 0.5, 0, 0])
            continue

        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except ValueError:
            pdb.set_trace()
    return new_array