import os, sys
from distutils.util import strtobool
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.util import nest, tf_inspect
from tensorflow.python.eager import tape
from tensorflow.python.ops.custom_gradient import graph_mode_decorator


# 是否使用重计算
do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))


def align(tensor, axes, ndim=None):
    """重新对齐tensor（批量版expand_dims）感觉更像是transpose
    axes: 原来的第i维对齐新tensor的第axes[i]维；
    ndim: 新tensor的维度
    """
    assert len(axes) == K.ndim(tensor)
    indices = [None] * (ndim or max(axes))
    for i in axes:
        # TODO(1、slice看笔记再确认，并且参考tf将examples写进注释)
        indices[i] = slice(None)
    return tensor[indices]


def sequence_masking(x, mask, value=0, axis=None):
    """为序列条件mask的函数

    parameters:
    -----------
    x: tensor
        输入张量
    mask: tensor
        形如(batch_size, seq_len)的0-1矩阵
    value: float or str
        mask部分要被替换成的值，允许'inf'与'-inf'
    axis: int
        序列所在的轴，默认为1
    """
    if mask is None:
        return x
    # 确保x类型，可以执行*运算
    x_type = K.dtype(x)
    if x_type == 'bool':
        x = K.cast(x, 'int32')
    # 确保mask类型 = x类型
    if K.dtype(mask) != K.dtype(x):
        mask = K.cast(mask, K.dtype(x))
    if value == '-inf':
        # -----------是个函数吗？？---------------
        value = -K.infinity
    if value == 'inf':
        value = K.infinity
    value = K.cast(value, K.dtype(x))
    # 确定axis
    if axis is None:
        axis = 1
    if axis < 0:
        axis = K.ndim(x) + axis
    assert axis > 0, 'axis must be greater than 0'
    # 统一shape
    for _ in range(axis - 1):  # > 1时生效
        mask = K.expand_dims(mask, 1)  # 把第0维让给batch_size
    for _ in range(K.ndim(x) - K.ndim(mask)):
        mask = K.expand_dims(mask, K.ndim(mask))
    x = x * mask + value * (1 - mask)
    # 与输入x的类型统一
    if x_type == 'bool':
        x = K.cast(x, x_type)
    return x


def recompute_grad(call):
    # ----------------------完全没看懂？？？？------------------------
    """重计算装饰器，用来装饰keras层的call函数
    目的是：通过一些额外的计算减少显存的占用
    论文：https://arxiv.org/abs/1604.06174
    """
    if not do_recompute:
        return call

    def inner(self, inputs, **kwargs):
        # 2.x的tf.nest.flatten不会对numpy和tf.tensor进行展平
        flat_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            """定义前向计算
            """
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            """定义前向计算和反向计算
            """
            with tape.stop_recording():
                outputs = kernel_call()
                outputs = tf.identity(outputs)

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(
                    outputs, watches, output_gradients=[doutputs]
                )
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        outputs, grad_fn = call_and_grad(*flat_inputs)
        flat_outputs = nest.flatten(outputs)

        def actual_grad_fn(*doutputs):
            grads = grad_fn(*doutputs, variables=self.trainable_weights)
            return grads[0] + grads[1]

        watches = flat_inputs + self.trainable_weights
        watches = [tf.convert_to_tensor(x) for x in watches]
        tape.record_operation(
            call.__name__, flat_outputs, watches, actual_grad_fn
        )
        return outputs

    return inner



def infinity():
    """返回默认的代表无穷大的数值
    """
    return tf.keras.utils.get_custom_objects().get('infinity', 1e12)


def set_infinity(value):
    """设置新的代表无穷大的数值
    """
    tf.keras.utils.get_custom_objects()['infinity'] = value


# 添加到 keras.backend 上，使其可以像 K.epsilon() 那样操作
K.infinity = infinity
K.set_infinity = set_infinity
sys.modules['tensorflow.keras.backend'] = K

