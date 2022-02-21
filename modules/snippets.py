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



if __name__ == '__main__':
    a = np.arange(32).reshape(1,2,4,4)
    b = orthogonally_resize(a, (1,2,6,6))
    print('b shape: ', b.shape)
    print('b: \n', b)

