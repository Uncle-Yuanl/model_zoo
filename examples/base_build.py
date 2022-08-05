import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from modules.models import *
from modules.layers import ConditionalRandomField

config_path = '/mnt/disk3/BERT预训练/retrain_2000w/9/bert_config.json'
checkpoint_path = '/mnt/disk3/BERT预训练/retrain_2000w/9/bert_model.ckpt'
dict_path = '/mnt/disk3/BERT预训练/retrain_2000w/9/config.json'


def test_position_embedding():
    pe = PositionEmbedding(100, 200)
    print(pe)


def test_model():
    model = build_transformer_model(
        config_path,
        checkpoint_path
    )

    # print('base transformer: \n', model.summary())

    bert_layers = 12
    output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
    output = model.get_layer(output_layer).output
    # 没有激活
    output = Dense(13)(output)
    CRF = ConditionalRandomField(100)
    output = CRF(output)
    print('crf output:\n', output)

    print('是否built：', model.built)
    model = Model(model.input, output)
    print('CRF transformer: \n', model.summary())

    model.compile(
        loss=CRF.sparse_loss,
        optimizer=tf.optimizers.Adam(1e-5),
        metrics=[CRF.sparse_accuracy]
    )
    print('是否built：', model.built)


if __name__ == '__main__':
    test_model()