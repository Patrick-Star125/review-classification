import re
import os
import mindspore
import argparse
from tqdm import tqdm
import mindspore.dataset as ms_dataset
import numpy as np
from mindspore import load_checkpoint, load_param_into_net
from mindspore import set_context, PYNATIVE_MODE
import math
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import Uniform, HeUniform

set_context(mode=PYNATIVE_MODE)


def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="train lstm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--rw_path', type=str, default='C:\\datasets\\aclImdb_v1.tar.gz', help='review dataset path')
    parser.add_argument('--weibo_path', type=str, default='C:\\datasets\\glove.6B.zip', help='weibo embedding path')
    parser.add_argument('--output_path', default='save_model/', type=str, help='the path model saved')
    parser.add_argument('--model_path', default='C:\\model\\sentiment-analysis.ckpt', type=str, help='model path')
    # 解析参数
    args_opt = parser.parse_args()
    return args_opt


def replace_all_blank(value):
    # \W 表示匹配非数字字母下划线
    result = re.sub('\W+', '', value).replace("_", '')
    return result


class IMDBData():
    """IMDB数据集加载器

    加载IMDB数据集并处理为一个Python迭代对象。

    """

    def __init__(self, path, mode="train"):
        self.mode = mode
        self.path = path
        self.docs, self.labels = [], []

        self._load(mode)

    def _load(self, label):
        if label == 'train':
            f_path = os.path.join(self.path, label, 'data.txt')
        else:
            f_path = os.path.join(self.path, label, 'test.txt')
        # 将数据加载至内存
        with open(f_path, mode='r', encoding='utf-8', errors='ignore') as f:
            line = f.readline()

            while line:
                # 对文本进行分词、去除标点和特殊字符、小写处理
                if label == 'train':
                    line = line.split(',', 1)
                    self.labels.append([int(line[0])])  # [[1]]
                    self.docs.append(list(replace_all_blank(line[1])))  # [['asd', 'fgh']]
                else:
                    self.labels.append([int(0)])
                    self.docs.append(list(replace_all_blank(line)))
                line = f.readline()

    def __getitem__(self, idx):
        return self.docs[idx], self.labels[idx]

    def __len__(self):
        return len(self.docs)


def load_weibo(weibo_path):
    embeddings = []
    tokens = []

    with open(weibo_path, encoding='utf-8') as gf:
        # 头一行内容没有用
        weibo = gf.readline()
        weibo = gf.readline()
        while weibo:
            word, embedding = weibo.split(sep=' ', maxsplit=1)
            tokens.append(word)
            embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))
            weibo = gf.readline()
    # 添加 <unk>, <pad> 两个特殊占位符对应的embedding
    embeddings.append(np.random.rand(300))
    embeddings.append(np.zeros((300,), np.float32))

    vocab = ms_dataset.text.Vocab.from_list(tokens, special_tokens=["<unk>", "<pad>"], special_first=False)
    embeddings = np.array(embeddings).astype(np.float32)
    return vocab, embeddings


class RNN(nn.Cell):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, embedding_table=Tensor(embeddings),
                                      padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.fc = nn.Dense(hidden_dim * 2, output_dim, weight_init=weight_init, bias_init=bias_init)
        self.dropout = nn.Dropout(1 - dropout)
        self.sigmoid = ops.Sigmoid()

    def construct(self, inputs):
        embedded = self.dropout(self.embedding(inputs))
        _, (hidden, _) = self.rnn(embedded)
        hidden = self.dropout(mnp.concatenate((hidden[-2, :, :], hidden[-1, :, :]), axis=1))
        output = self.fc(hidden)
        return self.sigmoid(output)


def data_preprocessing(vocab, rw_test):
    lookup_op = ms_dataset.text.Lookup(vocab, unknown_token='<unk>')
    pad_op = ms_dataset.transforms.c_transforms.PadEnd([500], pad_value=vocab.tokens_to_ids('<pad>'))
    type_cast_op = ms_dataset.transforms.c_transforms.TypeCast(mindspore.float32)

    rw_test = rw_test.map(operations=[lookup_op, pad_op], input_columns=['text'])
    rw_test = rw_test.map(operations=[type_cast_op], input_columns=['label'])

    rw_test = rw_test.batch(64, drop_remainder=False)

    return rw_test


def predict_sentiment(model, test_dataset, epoch=0):
    total = test_dataset.get_dataset_size()
    model.set_train(False)

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        path = os.path.join(os.getcwd(), 'result', 'result.txt')
        if os.path.exists(path):
            os.remove(path)
        for i in test_dataset.create_tuple_iterator():
            predictions = model(i[0])
            predictions = np.round(predictions.asnumpy())
            with open(path, mode='a', encoding='utf-8') as f:
                for j in predictions:
                    f.write(str(int(j)) + '\n')
            t.set_postfix()
            t.update(1)


def inference_process(args_opt):
    # load data
    rw_test = ms_dataset.GeneratorDataset(IMDBData(args_opt.rw_path, "test"), column_names=["text", "label"],
                                            shuffle=False)
    vocab, embeddings = load_weibo(args_opt.weibo_path)

    # check the embedding of word "the"
    idx = vocab.tokens_to_ids('the')
    embedding = embeddings[idx]
    print('the: ', embedding)

    # data preprocessing, adjust tensor shape
    rw_test = data_preprocessing(vocab, rw_test)

    # setting some hyper parameters
    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = vocab.tokens_to_ids('<pad>')

    # construct model
    net = RNN(embeddings, hidden_size, output_size, num_layers, bidirectional, dropout, pad_idx)

    # 将模型参数存入parameter的字典中
    param_dict = load_checkpoint(args_opt.model_path)

    # 将参数加载到网络中
    load_param_into_net(net, param_dict)
    # 生成测试结果
    predict_sentiment(net, rw_test)


if __name__ == '__main__':
    args_opt = parse_args()
    inference_process(args_opt)
