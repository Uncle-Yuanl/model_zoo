from modules.tokenizers import *

dict_path = '/mnt/disk3/BERT预训练/retrain_2000w/9/vocab.txt'


tokenizer = Tokenizer(dict_path, do_lower_case=True)


def test_encode(text):
    print('tokenize: ')
    print(tokenizer.encode(text, maxlen=256))


def test_tokenize(text):
    print('tokenize: ')
    print(tokenizer.tokenize(text, maxlen=256))


def test_rematch(text):
    tokens = tokenizer.tokenize(text, maxlen=256)
    print('tokens: ', tokens)
    print(tokenizer.rematch(text, tokens))


if __name__ == '__main__':
    string = """we're playing with the fire"""
    test_rematch(string)