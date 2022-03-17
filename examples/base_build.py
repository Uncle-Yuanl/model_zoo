import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from modules.models import *

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

    print(model.summary())

if __name__ == '__main__':
    test_model()