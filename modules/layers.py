import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import initializers, activations



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
        out_dim=None,  # ？？
        key_size=None,  # ？？
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
        if p_bias == 'rotary':
            # ----------------inputs的意义和维度，然后测试api--------------
            cos_pos = K.repeat_elements(inputs[n][..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(inputs[n][..., None, ::2], 2, -1)
            # ----------------还是先把论文看了吧---------------------------
            qw2 = K.stack()


custom_objects = {
    'Concatenate1D': Concatenate1D,
}

tf.keras.utils.get_custom_objects().update(custom_objects)


if __name__ == '__main__':
    pass