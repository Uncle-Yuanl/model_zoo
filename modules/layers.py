import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import initializers, activations

from modules.backend import sequence_masking, recompute_grad


# 父类不能直接是Layer，即时导入了
class Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.supports_masking = True  # 本项目的自定义层均可mask


class Concatenate1D(Layer):
    """1维序列拼接层

    避免有mask的序列与无mask序列拼接时出错
    """
    def call(self, inputs):
        return K.concatenate(inputs, axis=1)

    def compute_mask(self, inputs, mask=None):
        """还不太清楚这个mask是干嘛的？？？？
        """
        if mask is not None:
            masks = []
            for i, m in enumerate(mask):
                if m is None:
                    # batch中第i个样本的seq_len
                    m = K.ones_like(inputs[i][..., 0], dtype='bool')
                masks.append(m)
            return K.concatenate(masks, axis=1)

    def compute_output_shape(self, input_shape):
        # 所有tensor的维度都大于2？？？？
        if all([shape[1] for shape in input_shape]):
            seq_len = sum([shape[1] for shape in input_shape])
            # 插在seq_len， dim之间干嘛的？？？？
            return (input_shape[0][0], seq_len, input_shape[0][2])
        else:
            return (input_shape[0][0], None, input_shape[0][2])


class MultiHeadAttention(Layer):
    """多头注意力
    """
    def __init__(
        self,
        heads,
        head_size,  # hidden_dim / heads？？
        out_dim=None,
        key_size=None,  # q, k 的dimension
        use_bias=True,
        attention_scale=True,
        attention_dropout=None,
        return_attention_scores=False,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(initializers)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    # -------------待添加-----------------
    @recompute_grad
    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力

        q_mask: 对输入query序列的mask
                主要将输出结果的padding部分置0
        v_mask: 对输入value序列的mask
                主要防止attention读取到padding的信息
        """
        # ---------检查输入--------------------
        q, k,v = inputs[:3]
        q_mask, v_mask = None, None
        if mask is not None:
            # 强制规定如果输入则对应三个mask
            q_mask, v_mask = mask[0], mask[2]
        # 线型变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        # 即使self.key_size != head_size， seq_len一致
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        qkv_inputs = [qw, kw, vw] + inputs[3:]
        qv_mask = [q_mask, v_mask]
        o, a = self.pay_attention_to(qkv_inputs, qv_mask, **kwargs)
        # 完成输出
        o = K.reshape(o, (-1, K.shape(o)[1], self.head_size * self.heads))
        o = self.o_dense(o)
        # 返回结果
        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    def pay_attention_to(self, inputs, mask=None, **kwargs):
        """实现标准的乘性多头注意力
        单独实现可以方便子类定义不同形式的Attention

        a_bias: 对attention矩阵的bias。
                不同的attention bias对应不同的应用。
        p_bias: 在attention里的位置偏置。
                一般用来指定相对位置编码的种类。

        parameters:
        -----------
        inputs: list
            q, k, v + tensor
        mask: list
            q, v的mask

        returns:
        --------
        o: tensor
            shape = (batch_size, seq_len, heads, head_size)
        a: tensor
            shape = (
        """
        (qw, kw, vw), n = inputs[:3], 3
        q_mask, v_mask = mask
        a_bias, p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if a_bias:
            # ----------？？？？-----------
            a_bias = inputs[n]
            n += 1
        # TODO(学习position bias，包括绝对与相对)
        if p_bias == 'rotary':
            # ----------------inputs的意义和维度，然后测试api--------------
            cos_pos = K.repeat_elements(inputs[n][..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(inputs[n][..., None, ::2], 2, -1)
            # ----------------还是先把论文看了吧---------------------------
            qw2 = K.stack()
            # ...
            kw = kw * cos_pos + kw2 * sin_pos
        # Attention
        # d没了就是在d维度上累加，先将h转置到前面，然后批量地做('...jd,...dk->...,jk')
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        # --------------------两个都不懂？？-----------------------
        if p_bias == 'typical_relative':
            position_bias = inputs[n]
            # expand_dim
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, position_bias)
        if p_bias == 'T5_relative':
            position_bias = K.permute_dimensions(inputs[n], (2, 0, 1))
            # 加上batch_size的维度
            a = a + K.expand_dims(position_bias, 0)
        # Attention scale
        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None:
            a = a + a_bias
        a = sequence_masking(a, v_mask, '-inf', -1)
        A = K.softmax(a)
        if self.attention_dropout:
            A = Dropout(self.attention_dropout)(A)
        # 完成输出
        # 多一步转置，保证最后是(batch_size, seq_len, heads, head_size)
        o = tf.einsum('bhjk,bkhd->bjhd', A, vw)
        if p_bias == 'typical_relative':
            o = o + tf.einsum('bhjk,jkd->bjhd', A, position_bias)
        return o, a

    def comput_output_shape(self, input_shape):
        # -------------------确定下input_shape？？？？-------------------------------------
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            # ---------------这里heads又在前面了-----------------------
            a_shape = (input_shape[0][0], self.heads, input_shape[0][1],input_shape[1][1])
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                # -------------------干嘛用的，返回的应该是q_mask吧？？---------------------
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))





custom_objects = {
    'Concatenate1D': Concatenate1D,
}

tf.keras.utils.get_custom_objects().update(custom_objects)


if __name__ == '__main__':
    pass