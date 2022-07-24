from collections import defaultdict


class Vocab():
    def __init__(self, token2id_dic, id2token_dic):

        self.token2id = token2id_dic
        self.id2token = id2token_dic

    def sentence2ids(self,sentence):
        return [self.token2id[token] for token in sentence]

    def ids2sentence(self,ids):
        sentence =  [self.id2token[id] for id in ids]
        return " ".join(sentence)

    def add_token(self,token):
        assert token not in self.token2id.keys()
        self.token2id[token] = len(self.token2id)
        self.id2token[len(self.token2id)] = token

    @classmethod
    def load_vocab(cls, vocab_path):
        token2id_dic = cls._init_token()
        with open(vocab_path,encoding="utf-8") as f:
            for token in f.readlines():
                n = len(token2id_dic)
                token2id_dic[token.strip()] = n
        id2token_dic = list(token2id_dic.keys())

        return cls(token2id_dic, id2token_dic)

    @staticmethod
    def _init_token():
        vocab = {
            "<pad>": 0,
            "<mask>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        return vocab

    @staticmethod
    def from_data_get_vocab(data_path, vocab_path, min_freq=-1, max_token_num=None):
        '''

        :param data_path: 数据路径
        :param freqs: 加入词表最低频次
        :param max_token_num: 词典大小
        :return:
        '''
        vocab = defaultdict(int)
        max_freq = 0
        with open(data_path,encoding="utf-8") as f:
            sentences = f.readlines()
            for sentence in sentences:
                sentence = sentence.strip().split()
                for token in sentence:
                    vocab[token] += 1
                    if vocab[token] > max_freq:
                        max_freq = vocab[token]
            vocab = sorted(vocab.items(), key=lambda x: x[1],reverse=True)
        with open(vocab_path, 'w',encoding="utf-8") as f:
            for item in vocab[:max_token_num]:
                f.write(item[0] + '\n')

    def __len__(self):
        return len(self.token2id)
