import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from .layers import *


class Transformer(object):
    """模型基类
    """
    def __init__(
        self,
        vocab_size,  # 词表大小
        hidden_size,  # 编码维度
        num_hidden_layers,  # Transformer总层数
        num_attention_heads,  # Attention的头数
        intermediate_size,  # FeedForward的隐层维度
        hidden_act,  # FeedForward隐层的激活函数
        dropout_rate=None,  # Dropout比例
        attention_dropout_rate=None,  # Attention矩阵的Dropout比例？？？？
        embedding_size=None,  # 是否指定embedding_size  # 指定了会怎么样，裁剪 or padding？？
        attention_head_size=None,  # Attention中V的head_size
        attention_key_size=None,  # Attention中Q,K的head_size
        sequence_length=None,  # 是否固定序列长度  # 还有不固定的吗，怎么实现？？
        keep_tokens=None,  # 要保留的词ID列表  # 干啥的？？  # embeddings矩阵根据ID选择并重构
        compound_tokens=None,  # 扩展Embedding
        residual_attention_scores=False,  # Attention矩阵加残差  # 为什么要有，是有论文吗？？  # 有
        ignore_invalid_weights=False,  # 允许跳过不存在的权重  # 如何跳过，然后随机初始化吗？？
        autoresize_weights=False,  # 自动变换形状不匹配的权重
        layers=None,  # 外部传入的Keras层
        prefix=None,  # 层名前缀
        name=None,  # 模型名称
        **kwargs
    ):
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size  # 这里看出维度并没有完全要求一致？？
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_dropout_rate = attention_dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights  #
        self.autoresize_weights = autoresize_weights
        self.layers = {} if layers is None else layers
        self.prefix = prefix or ''
        self.name = name
        self.built = False  # 表示模型是否已经完成build

    def build(
        self,
        attention_caches=None,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs
    ):
        """模型构建函数

        parameters:
        -----------
        attention_caches: dict
            为Attention的k, v的缓存序列字段，{Attention层名: [k缓存, v缓存]}
        layer_norm_*: ??
            实现Conditional Layer Normalization时使用，用来实现"固定长度向量"为条件的条件Bert
            啥呀？？？？
        """
        if self.built:
            return None
        # Input
        inputs = self.get_inputs()
        self.set_inputs(inputs, additional_input_layers)
        # Other
        self.attention_caches = attention_caches or {}
        self.layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or 'linear',
        ]
        # Call
        outputs = self.call(inputs)
        self.set_outputs(outputs)
        # Model
        self.model = Model(self.inputs, self.outputs, name=self.name)
        self.built = True

    def call(self, *inputs):
        """定义模型执行流程

        这种方法将层与结构解耦，方便子类配置
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        for i in range(self.num_hidden_layers):
           outputs = self.apply_main_layers(outputs, i)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

    def prefixed(self, name):
        """给名字加前缀
        """
        if name is not None:
            return self.prefix + name

    def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
        """通过apply调用层会自动重用同名层

        parameters:
        -----------
        inputs: tensor
            上一层的输出
        layer: class
            要调用的层类名？？？？ 哪里导入的？？？？
        arguments:???
            传递给layer.call的参数
        kwargs: dict
            传递给层初始化的参数
        """
        if layer is Dropout and self.dropout_rate == 0:
            return inputs

        if layer is MultiHeadAttention and self.residual_attention_scores:
            kwargs['return_attention_scores'] = True

        arguments = arguments or {}
        if layer is Lambda:
            kwargs['arguments'] = arguments
            arguments = {}

        name = self.prefixed(kwargs.get('name'))
        kwargs['name'] = name
        # 如果name = None，都是False
        if name not in self.layers:  # 外部传入的keras层
            layer = layer(**kwargs)  # 实例化
            name = layer.name
            self.layers[name] = layer  # 将层名与层实例映射

        if inputs is None:
            return self.layers[name]
        else:
            if isinstance(self.layers[name], MultiHeadAttention):
                if name in self.attention_caches:
                    # 如果检测到Cache的传入，name自动在Key，Value处拼接起来
                    # 为什么只有两个，q没有？？？？
                    k_cache, v_cache = self.attention_caches[name]
                    k_name, v_name = name + '-Cached-Key', '-Cached-Value'
                    # 已测，可以调用
                    k = Concatenate1D(name=k_name)([k_cache, inputs[1]])
                    v = Concatenate1D(name=v_name)([v_cache, inputs[2]])
                    inputs = inputs[:1] + [k, v] + inputs[3:]
                if self.residual_attention_scores:
                    # 如何使用残差Attention矩阵，则给每个Attention矩阵加上前上一层的Attention
                    # 矩阵，这对应RealFormer设计（https://arxiv.org/abs/2012.11747）。
                    # 目前该实现还相对粗糙，可能欠缺通用性。
                    if self.attention_scores is not None:
                        # 看一下arguments在call函数中是怎么起作用的？？？？
                        if arguments.get('a_bias'):
                            # 所以这个inputs[3]是什么，看下jupyter的笔记
                            a_bias = Add(name=name + '-Attention-Bias')(inputs[3], self.attention_scores)
                            inputs = inputs[:3] + [a_bias] + inputs[3:]
                        else:
                            a_bias = self.attention_scores
                            inputs = inputs[:3] + [a_bias] + inputs[4:]
                        arguments['a_bias'] = True
                    o, a = self.layers[name](inputs, **arguments)
                    # 初始/更新残差
                    self.attention_scores = a
                    return o
            return self.layers[name](inputs, **arguments)

    def get_inputs(self):
        raise NotImplementedError

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs, index):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def compute_attention_bias(self, inputs=None):
        """定义每次层的Attention Bias
        """
        return self.attention_bias

    def compute_position_bias(self, inputs=None):
        """定义每次层的P哦四通 Bias (一般相对位置编码用)
        """
        return self.position_bias

    def set_inputs(self, inputs, additional_input_layers=None):
        """设置input和inputs属性
        """
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        # deepcopy
        inputs = inputs[:]
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers, list):
                additional_input_layers = [additional_input_layers]
            inputs.extend(additional_input_layers)

        self.inputs = inputs
        # -------------------提取第一个input的话是不是写错了？---------------------------
        if len(inputs) > 1:
            self.input = inputs
        else:
            # 这是干嘛？？保持维度吗？？
            self.input = inputs[0]

    def set_outputs(self, outputs):
        """设置output和outputs属性
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(stddev=0.02)

    def simplify(self, inputs):
        """将list中的None过滤掉

        干嘛用的？？？？
        """
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            # 这里又符合正常逻辑了
            inputs = inputs[0]

        return inputs

    def load_embeddings(self, embeddings):
        """处理Embedding层权重
        """
        # -------------embeddings是数组吗？？--------------
        embeddings = embeddings.astype(K.floatx())

        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                if isinstance(item, list):
                    # 权重weights都是1
                    item = (item, [1] * len(item))
                ext_embeddings.append(np.average(embeddings[item[0]], 0, item[1]))
            embeddings = np.concatenate([embeddings, ext_embeddings], 0)

        return embeddings

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        if isinstance(checkpoint, dict):
            return checkpoint[name]
        else:
            return tf.train.load_variable(checkpoint, name)

    def create_variable(self, name, value, dtype=None):
        """创建一个变量
        """
        dtype = dtype or K.floatx()
        return K.variable(
            # 这里不是加了装饰器？应该不能再执行__call__()？？
            self.initializer(value.shape, dtype), dtype, name
        ), value

    def variable_mapping(self):
        """建立keras层与checkpoint的变量名之间的映射
        """
        return {}

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        # -----------------这个mapping是在哪里完成配置的？？------------------------
        mapping = mapping or self.variable_mapping()
        # 加上前缀
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        # 只保留self.layers中的层
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        weight_value_pairs = []
        for layer, variables in mapping.items():
            # layer: 层名， variables: checkpoint变量名
            # ---------------------所以self.layers是怎么完成配置的？？----------------------------
            layer = self.layers[layer]
            weights, values = [], []

            for w, v in zip(layer.trainable_weights, variables):  # 允许跳过不存在的权重
                try:
                    values.append(self.load_variable(checkpoint, v))
                    weights.append(w)
                except Exception as e:
                    if self.ignore_invalid_weights:
                        print('%s, but ignored.' % e.message)
                    else:
                        raise e

            for i, (w, v) in enumerate(zip(weights, values)):
                if v is not None:
                    w_shape, v_shape = K.int_shape(w), v.shape
                    if self.autoresize_weights and w_shape != v_shape:
                        # ----------------为什么是对v进行操作-------------------
                        v = orthogonally_resize(v, w_shape)
                        if isinstance(layer, MultiHeadAttention):
                            # -------------这个count是啥----------------
                            count = 2
                            if layer.use_bias:
                                count += 2
                            if layer.attention_scale and i < count:
                                scale = 1.0 * w_shape[-1] / v_shape[-1]
                                # 这里咋就4次方根
                                v = v * scale**0.25
                            if isinstance(layer, FeedForward):
                                count = 1
                                if layer.use_bias:
                                    count += 1
                                if self.hidden_act in ['relu', 'leaky_relu']:
                                    count -= 2
                                if i < count:
                                    v *= np.sqrt(1.0 * w_shape[-1] / v_shape[-1])
                                else:
                                    # ----------------为什么这里选择第0维度--------------
                                    v *= np.sqrt((1.0 * w_shape[0]) / v_shape[0])

                    weight_value_pairs.append((w, v))
        # ----------API学习，set完了模型的权重就更新了吗？？---------------
        K.batch_set_value(weight_value_pairs)

    def save_weights_as_checkpoint(self, filename, mapping=None, dtype=None):
        """根据mapping将权重保存为checkpoint格式
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            all_variables, all_values = [], []
            for layer, variables in mapping.items():
                layer = self.layers[layer]
                values = K.batch_get_value(layer.trainable_weights)
                for name, value in zip(variables, values):
                    variable, value = self.create_variable(name, value, dtype)
                    all_variables.append(variable)
                    all_values.append(value)
            # tensorflow 2.x 貌似已经没有了
            # 直接使用tf.train.CheckpointManager以及Checkpoint类
            with tf.Session() as sess:
                K.batch_set_value(zip(all_variables, all_values))
                saver = tf.train.Saver()
                saver.save(sess, filename)3














