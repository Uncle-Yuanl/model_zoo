import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import initializers, activations

from modules.backend import sequence_masking, recompute_grad


def integerize_shape(func):
    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func


# 父类不能直接是Layer，即时导入了
class Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.supports_masking = True  # 本项目的自定义层均可mask


# TODO(这样condition)
class ScaleOffset(Layer):
    """简单的仿射变换层（最后一维乘上gamma向量并加上beta向量）
    说明：1、具体操作为最后一维乘上gamma向量，并加上beta向量；
         2、如果直接指定scale和offset，那么直接常数缩放和平移；
         3、hidden_*系列参数仅为有条件输入时（conditional=True）使用，
            用于通过外部条件控制beta和gamma。
    """
    def __init__(
        self,
        scale=True,
        offset=True,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs
    ):
        super(ScaleOffset, self).__init__(**kwargs)
        self.scale = scale
        self.offset = offset
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(ScaleOffset, self).build(input_shape)

        # TODO(为什么有conditional的区别)
        if self.conditional:
            input_shape = input_shape[0]

        if self.offset is True:
            # TODO(1、keras.layers.Layer的类方法)
            self.beta = self.add_weight(
                name='beta', shape=input_shape[-1], initializer='zeros'
            )
        if self.scale is True:
            self.gamma = self.add_weight(
                name='gamma', shape=input_shape[-1], initializer='ones'
            )
        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.offset is not False and self.offset is not None:
                self.beta_dense = Dense(
                    units=input_shape[-1],
                    use_bias=False,
                    kernel_initializer='zeros'
                )
            if self.scale is not False and self.scale is not None:
                self.gamma_dense = Dense(
                    units=input_shape[-1],
                    use_bias=False,
                    # ---------为啥又是0了？？--------
                    kernel_initializer='zeros'
                )

    def compute_mask(self, inputs, mask=None):
        # tf2.7源码：https://github.com/keras-team/keras/blob/v2.8.0/keras/engine/base_layer.py#L959-L979
        # 区别是自定义的Layer都支持mask
        if self.conditional:
            return mask if mask is None else mask[0]
        else:
            return mask

    @recompute_grad
    def call(self, inputs):
        if self.conditional:
            # TODO(conds到底是什么？？同CLN)
            inputs, conds = inputs
            if self.hidden_units is not None:
                conds = self.hidden_dense(conds)
            conds = align(conds, [0, -1], K.ndim(inputs))

        if self.scale is not False and self.scale is not None:
            # 允许直接传入scale作为gamma向量
            gamma = self.gamma if self.scale is True else self.scale
            if self.conditional:
                gamma = gamma + self.gamma_dense(conds)
            inputs = inputs * gamma

        if self.offset is not False and self.offset is not None:
            beta = self.beta if self.offset is True else self.offset
            if self.conditional:
                beta = beta + self.beta_dense(conds)
            inputs = inputs + beta

        return inputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'scale': self.scale,
            'offset': self.offset,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer': initializers.serialize(self.hidden_initializer),
        }
        base_config = super(ScaleOffset, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
        self.kernel_initializer = initializers.get(kernel_initializer)

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
        # --------------------看input[3:]的数据维度含义-----------------------
        if a_bias:
            a_bias = inputs[n]
            n += 1
        # TODO(学习position bias，包括绝对与相对)
        if p_bias == 'rotary':
            # 公式链接：https://spaces.ac.cn/archives/8265
            # inputs[n]是Sinusoidal位置编码，θi=10000^(−2i/d)
            cos_pos = K.repeat_elements(inputs[n][..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(inputs[n][..., None, ::2], 2, -1)
            # 为了缓解稀疏空间浪费问题，采用两个列向量(q * S)pairwise相乘，在相加
            qw2 = K.stack(-qw[..., 1::2], qw[..., ::2], 4)
            qw2 = K.reshape(qw2, K.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = K.stack(-kw[..., 1::2], kw[..., ::2], 4)
            kw2 = K.reshape(kw2, K.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
        # Attention
        # d没了就是在d维度上累加，先将h转置到前面，然后批量地做('...jd,...dk->...,jk')
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        if p_bias == 'typical_relative':
            position_bias = inputs[n]
            #          WQ * Rij^T，公式中的xj被加和掉了。注意head紧跟在batch_size后面
            # 注意position_bias没有batch_size维度，T5也是，那后面的k是啥
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, position_bias)
        if p_bias == 'T5_relative':
            position_bias = K.permute_dimensions(inputs[n], (2, 0, 1))
            # 加上batch_size的维度，广播？？？？
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


# TODO(为什么会继承ScaleOffset？？也就是看call函数最后的super)
class LayerNormalization(ScaleOffset):
    """(cnditional) Layer Normalization
    """
    def __init__(
        self,
        zero_mean=True,
        unit_variance=True,
        epsilon=None,
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.zero_mean = zero_mean
        self.unit_variance = unit_variance
        self.epsilon = epsilon or 1e-12

    @recompute_grad
    def call(self, inputs):
        """如果是Conditional LayerNormalization，则默认以list为输入，第二个是条件
        """
        if self.conditional:
            inputs, conds = inputs

        if self.zero_mean:
            mean = K.mean(inputs, axis=-1, keepdims=True)
            inputs = inputs - mean
        if self.unit_variance:
            variance = K.mean(K.square(inputs), axis=-1, keepdims=True)
            inputs = inputs / K.sqrt(variance + self.epsilon)

        if self.conditional:
            inputs = [inputs, conds]

        return super(LayerNormalization, self).call(inputs)

    def get_config(self):
        config = {
            'zero_mean': self.zero_mean,
            'unit_variance': self.unit_variance
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PositionEmbedding(Layer):
    """定义可训练的位置Embedding
    """
    def __int__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        hierarchical=None,
        embeddings_initializer='zeros',
        custom_position_ids=False,
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embedding',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'int' not in K.type(position_ids):
                position_ids = K.cast(position_ids, 'int32')
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            # [None]: (seq_len,) --> (1, seq_len)，广播
            position_ids = K.arange(0, seq_len, dtype='int32')[None]

        if self.hierarchical:
            # TODO(看科学空间，并记录笔记在onenote)
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = K.gather(embeddings, position_ids // self.input_dim)
            embeddings_y = K.gather(embeddings, position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = K.gather(self.embeddings, position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(Layer):
    """FeedForward层
    如果activation不是一个list，那么它就是两个Dense层叠加；否则第一个Dense层将会被替换成门控线性单元（Gated Linear Unit）
    TODO(论文：https://arxiv.org/abs/2002.05202)
    """
    def __init__(
        self,
        units,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        if not isinstance(activation, list):
            activation = [activation]
        self.activation = [activations.get(act) for act in activation]
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        for i, activation in self.activation:
            i_dense = Dense(
                units=self.units,
                activation=activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            setattr(self, 'i%s_dense' % i, i_dense)

        self.o_dense = Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs):
        x = self.i0_dense(inputs)
        for i in range(1, len(self.activation)):
            # TODO(这里看下论文)
            x = x * getattr(self, 'i%s_dense' % i)(inputs)
        x = self.o_dense(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': [
                activations.deserialize(act) for act in self.activation
            ],
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.deserialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




custom_objects = {
    'Concatenate1D': Concatenate1D,
}

tf.keras.utils.get_custom_objects().update(custom_objects)


if __name__ == '__main__':
    pass