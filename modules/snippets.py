import os, sys, six, re, json
import unicodedata
import logging
import numpy as np
from collections import defaultdict

from modules.backend import K, tf

_open_ = open
is_py2 = six.PY2

if not is_py2:
    basestring = str


def is_string(s):
    """判断是否为字符串
    """
    return isinstance(s, basestring)


def lowercase_and_normalize(text):
    """转小写并进行简单的标准化
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    # Mn: Mark, Nonspacing
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text


def truncate_sequences(maxlen, indices, *sequences):
    """截断总长度至不超过maxlen
    循环地pop掉除了_token_end的最后一个元素

    parameters:
    indices: int
        pop的索引，一般为-1，如果加上了_token_end（如'[SEP]'）则为-2
    """
    # tuple -> list
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


# TODO(复习)
def orthogonally_resize(a, new_shape, window=2):
    """简单的正交化缩放矩阵

    parameters:
    -----------
    window: int
        控制复制时的批大小，不同维度同理
        例如：window = 2, 4 -> 6 ==> [0,1,2,3] -复制-> [0,1,0,1,2,3,2,3] -截断-> [0,1,0,1,2,3]
             window = 2, 4 -> 6 ==> [0,1,2,3] -复制-> [0,1,2,3,0,1,2,3] -截断-> [0,1,2,3,0,1]
    """
    assert a.ndim == len(new_shape)
    slices, a_norm, w = [], np.linalg.norm(a), window
    print(a.shape)
    print(new_shape)
    for i, (d1, d2) in enumerate(zip(a.shape, new_shape)):
        print('d1: ', d1, 'd2: ', d2)
        if d1 != d2:
            print('hit')
            k = d2 // d1 + int(d2 % d1 != 0)
            if k > 1:  # d2 > d1
                # 强制约定d1是window的整倍数，否则无法reshape
                assert d1 % window == 0
                a = a.reshape(a.shape[:i] + (d1 // w, w) + a.shape[i + 1:])
                # 先冗余复制
                a = np.repeat(a, k, axis=i)
                a = a.reshape(a.shape[:i] + (d1 * k, ) + a.shape[i + 2:])
        slices.append(np.s_[:d2])
    # 然后用new_shape切片
    a = a[tuple(slices)]
    # 与原范数相等
    return a / np.linalg.norm(a) * a_norm


class ViterbiDecoder:
    """Viterbi解码算法基类
    """
    def __init__(self, trans, starts=None, ends=None):
        self.trans = trans
        self.num_labels = len(trans)
        self.non_starts = []
        self.non_ends = []
        if starts is not None:
            for i in range(self.num_labels):
                if i not in starts:
                    self.non_starts.append(i)
        if ends is not None:
            for i in range(self.num_labels):
                if i not in ends:
                    self.non_ends.append(i)

    def decode(self, nodes):
        """nodes.shape=[seq_len, num_labels]
        """
        # 预处理，主要是[CLS], [SEP]固定为0
        nodes[0, self.non_starts] -= np.inf
        nodes[0, self.non_ends] -= np.inf

        # 动态规划
        labels = np.arange(self.num_labels).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        path = labels
        for l in range(1, len(nodes)):
            M = scores + self.trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(axis=0)
            scores = M.max(axis=0).reshape((-1, 1))
            # 注意path[:, idx]这种切片方式，按照idxs取0维上最后一个元素
            # [[0,0,0], [4,5,6]][:, [1,0,2]] -> [[0,0,0], [5,4,6]]
            # [[0,0,0], [4,5,6]][:, [1,0,0]] -> [[0,0,0], [5,4,4]]
            path = np.concatenate([path[:, idxs], labels], axis=0)

        # 最优路径
        return path[:, scores[:, 0].argmax()]


if __name__ == '__main__':
    a = np.arange(32).reshape(1,2,4,4)
    b = orthogonally_resize(a, (1,2,6,6))
    print('b shape: ', b.shape)
    print('b: \n', b)

