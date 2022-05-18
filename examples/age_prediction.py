import pandas as pd
from sklearn.utils import shuffle

from modules.tokenizers import Tokenizer
from modules.snippets import DataGenerator
from modules.models import build_transformer_model

from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback


def preprocess(
        content_path=r'F:\Jupyter_Note\Marcpoint\人群画像\年龄预测\data\content数据.csv',
        label_path=r'F:\Jupyter_Note\Marcpoint\人群画像\年龄预测\data\label数据.csv',
        seqlen=200):
    # x
    dfcontent = pd.read_csv(content_path, sep='\t')
    unique_id = dfcontent.groupby(by=['id']).apply(lambda x: ''.join(x['content']))
    unique_id = pd.DataFrame({
        'id': unique_id.index,
        'content': unique_id.values
    })
    unique_id['content_cut'] = unique_id['content'].apply(lambda x: x[:seqlen])
    # y
    dflabels = pd.read_csv(label_path, sep='\t')
    data = pd.merge(unique_id, dflabels, on=['id'])
    agemap = {
        '<15': 0,
        '15-20': 1,
        '21-25': 2,
        '26-30': 3,
        '31-35': 4,
        '36-40': 5,
        '41-50': 6,
        '51-60': 7,
        '61+': 8
    }
    data['user_age_label'] = data['user_age'].map(agemap)
    return data[['content_cut', 'user_age_label']]


num_classes = 9
maxlen = 200
batch_size = 10
learning_rate = 2e-5
epochs = 20

# BERT base
config_path = r'F:\Jupyter_Note\Marcpoint\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'F:\Jupyter_Note\Marcpoint\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'F:\Jupyter_Note\Marcpoint\chinese_L-12_H-768_A-12\vocab.txt'


def load_data(df):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    for idx, row in df.iterrows():
        D.append((row['content_cut'], int(row['user_age_label'])))
    return D


valid_ratio = 0.2
total_data = shuffle(load_data(preprocess(seqlen=maxlen)))
cut = int(len(total_data) * (1 - valid_ratio))
train_data = total_data[:cut]
valid_data = total_data[cut:]
print('train: ', len(train_data))
print('valid: ', len(valid_data))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            yield [token_ids, segment_ids], [[label]]  # 返回一条样本


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


# 加载模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)
print('bert输出类型：', type(bert.model.output))
print('bert输出维度：', bert.model.output.shape)
output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer,
    name='Probas'
)(output)

model = Model(bert.model.input, output)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['sparse_categorical_accuracy'],
)
model.summary()
bert.load_weights_from_checkpoint(checkpoint_path)


# 回调，保存最优模型
class Evaluator(Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs['sparse_categorical_accuracy']
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(r'E:\Models\age_prediction\best_model_{:.5f}.weights'.format(self.best_val_acc))
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def main_train():
    evaluator = Evaluator()

    train_dataset = train_generator.to_dataset(
        types=[('float32', 'float32'), ('float32',)],
        shapes=[([None], [None]), ([1],)],  # 配合后面的padded_batch=True，实现自动padding
        names=[('Input-Token', 'Input-Segment'), ('Probas',)],
        padded_batch=True
    )  # 数据要转为tf.data.Dataset格式，names跟输入层/输出层的名字对应

    valid_dataset = valid_generator.to_dataset(
        types=[('float32', 'float32'), ('float32',)],
        shapes=[([None], [None]), ([1],)],  # 配合后面的padded_batch=True，实现自动padding
        names=[('Input-Token', 'Input-Segment'), ('Probas',)],
        padded_batch=True
    )

    model.fit(
        train_dataset,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=valid_dataset,
        validation_steps=len(valid_generator),
        callbacks=[evaluator],
        verbose=1
    )


def main_prediction()



if __name__ == '__main__':
    # main_train()

