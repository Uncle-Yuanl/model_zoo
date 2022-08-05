import unicodedata, re
from modules.snippets import is_string
from modules.snippets import truncate_sequences
from modules.snippets import lowercase_and_normalize



def load_vocab(dict_path, encoding='utf-8', simplified=False, startswith=None):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            # 有不可见字符，也有空行（344行）
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    # 过滤冗余部分token
    if simplified:
        new_token_dict, keep_tokens = {}, []
        startswith = startswith or []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict and not Tokenizer._is_redundant(t):
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict


def save_vocab(dict_path, token_dict, encoding='utf-8'):
    """将词典（如精简过的）保存为文件
    """
    with open(dict_path, 'w', encoding=encoding) as writer:
        for k, v in sorted(token_dict.items(), key=lambda s: s[1]):
            writer.write(k + '\n')


class TokenizerBase:
    """分词器基类
    """
    def __init__(
        self,
        token_start='[CLS]',
        token_end='[SEP]',
        pre_tokenize=None,
        token_translate=None
    ):
        """参数说明：
        pre_tokenize：外部传入的分词函数，用作对文本进行预分词，如果传入
                      pre_tokenize，则先执行pre_tokenize(text)，然后在它
                      的基础上执行原来的tokenize函数；
        token_translate：映射字典，在tokenize之后，将某些特殊的token
                         替换为对应的token。
        """
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = token_start
        self._token_end = token_end
        self._pre_tokenize = pre_tokenize
        self._token_translate = token_translate or {}
        self._token_translate_inv = {
            v: k for k, v in self._token_translate.items()
        }

    def tokenize(self, text, maxlen=None):
        """分词函数
        """
        tokens = [
            self._token_translate.get(token) or token
            for token in self._tokenize(text)
        ]
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if maxlen is not None:
            # 留位置给_token_end
            index = int(self._token_end is not None) + 1
            truncate_sequences(maxlen, -index, tokens)

        return tokens

    def token_to_id(self, token):
        """token转为对应的id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        """token序列转换成对应的id序列
        """
        return [self.token_to_id(token) for token in tokens]

    def encode(
        self,
        first_text,
        second_text=None,
        maxlen=None,
        pattern='S*E*E',
        truncate_from='right'
    ):
        """输出文本对应的token id和segment id
        """
        if is_string(first_text):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text

        if maxlen:
            if truncate_from == 'right':
                index = -int(self._token_end is not None) - 1
            elif truncate_from == 'left':
                index =  int(self._token_start is not None)
            else:
                index = truncate_from
            if second_text and pattern == 'S*E*E':
                maxlen += 1
            truncate_sequences(maxlen, index, first_tokens, second_tokens)

        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def id_to_token(self, i):
        """id转为对应token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id序列转为对应的token序列
        """
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids):
        """转为可读文本
        """
        raise NotImplementedError

    def _tokenize(self, text):
        """基本分词函数
        """
        raise NotImplementedError


class Tokenizer(TokenizerBase):
    """Bert原生分词器
    """
    def __init__(
        self, token_dict, do_lower_case=True, word_maxlen=200, **kwargs
    ):
        super(Tokenizer, self).__init__(**kwargs)
        if is_string(token_dict):
            token_dict = load_vocab(token_dict)

        self._do_lower_case = do_lower_case
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self._vocab_size = len(token_dict)
        self._word_maxlen = word_maxlen

        # 补充特殊字符的id
        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass

    def token_to_id(self, token):
        """token转为对应的id
        """
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i):
        """id转成对应的token
        """
        return self._token_dict_inv[i]

    def decode(self, ids, tokens=None):
        """转为可读文本
        """
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text, flag = '', False
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif len(token) > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +', ' ', text)
        # 解决's类问题，注意左右都有空格
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)  # 去标点后一个空格
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)   # 去小数点后一个空格

        return text.strip()

    def _tokenize(self, text, pre_tokenize=True):
        """基本分词函数
        """
        if self._do_lower_case:
            text = lowercase_and_normalize(text)

        if pre_tokenize and self._pre_tokenize is not None:
            tokens = []
            for token in self._pre_tokenize(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend(self._tokenize(text, False))
            return tokens

        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self, word):
        """word内分成subword
        """
        if len(word) > self._word_maxlen:
            return [word]

        tokens, start, end = [], 0, 0
        while start < len(word):
            end = len(word)
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                end -= 1
            if start == end:
                return [word]
            else:
                tokens.append(sub)
                start = end

        return tokens

    @staticmethod
    def stem(token):
        """获取token的词干，如果是##开头，则自动去掉##
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch =='\t' or \
            unicodedata.category(ch) == 'Zs'  # Separator, Space

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        参考ASCII码表：https://www.habaijian.com/
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

    @staticmethod
    def _is_cjk_punctuation():
        """中文标点
        等价zhon.hanzi.punctuation
        """
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """C（Chinese）J（Japanese）K（Korean）类字符判定（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_spcecial(ch):
        """判断是否有特殊含义的符号
        主要判断'[CLS]'，'[SEP]'这类
        """
        return bool(ch) and (ch[0] == '[' and ch[-1] == ']')

    @staticmethod
    def _is_redundant(token):
        """判断该token是否冗余（默认情况下不可能分出来）
        冗余：词干为CJK字符或者标点符号
        """
        if len(token) > 1:
            for ch in Tokenizer.stem(token):
                if (
                    Tokenizer._is_cjk_character(ch) or
                    Tokenizer._is_punctuation(ch)
                ):
                    return True

    def rematch(self, text, tokens):
        """给出原始text和tokenize后tokens之间的映射关系
        TODO(测试)
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                print('before', ch)
                ch = lowercase_and_normalize(ch)
                print('after', ch)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            if len(ch) == 0:
                print(i)
                print(ch)
            char_mapping.extend([i] * len(ch))
        print(char_mapping)

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_spcecial(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = start

        return token_mapping







if __name__ == '__main__':
    pass












