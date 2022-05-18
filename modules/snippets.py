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


def to_array(*args):
    """批量转numpy的array
    """
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results


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


class DataGenerator:
    """数据生成器模板
    """
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or self.batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        else:
                            if len(caches) == self.buffer_size:
                                isfull = True
                    # isfull=False, data总量小于buffer_size
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            # hasattr(self.data, '__len__') = True
            else:
                def generator():
                    # 不重复乱序
                    for i in np.random.permutation(len(self.data)):
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        # TODO(如果没有__next__方法呢)
        d_current = next(data)  # 先取一个 后面开始迭代
        for d_next in data:
            yield False, d_current
            d_current = d_next
        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self, random=True):
        # TODO(怎么停？？？？)
        while True:
            for d in self.__iter__(random):
                yield d

    def fortest(self, random=False):
        while True:
            for d in self.__iter__(random):
                yield d[0]

    def to_dataset(self, types, shapes, names=None, padded_batch=False):
        """转为tf.data.Dataset格式
        如果传入names的话，自动把数据包装成dict形式。
        """
        if names is None:
            generator = self.forfit
        else:
            if is_string(names):
                warps = lambda k, v: {k, v}
            elif is_string(names[0]):
                warps = lambda k, v: dict(zip(k, v))
            else:
                # k, v本身是多维的，结果就第0维上每个元素，搞个dict
                # example：tuple(dict(zip(i, j)) for i, j in
                #  zip([[1,2,3], [3,4]],
                #     [['a', 'b', 'c'], ['c', 'd']]))
                # ({1: 'a', 2: 'b', 3: 'c'}, {3: 'c', 4: 'd'})
                warps = lambda k, v: tuple(
                    dict(zip(i, j)) for i, j in zip(k, v)
                )

            def generator():
                for d in self.forfit():
                    yield warps(names, d)

            types = warps(names, types)
            shapes = warps(names, shapes)

        if padded_batch:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types
            )
            dataset = dataset.padded_batch(self.batch_size, shapes)

        else:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types, output_shapes=shapes
            )
            dataset = dataset.batch(self.batch_size)

        return dataset


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

