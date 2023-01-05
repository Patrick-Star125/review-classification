# --rw_path C:\Users\NetPunk\Desktop\dataset\aclImdb_v1.tar.gz --weibo_path C:\Users\NetPunk\Desktop\dataset\glove.6B
# --rw_path C:\Users\NetPunk\Desktop\LSTM-TakeAway-Review\mindcon_text_classification --weibo_path C:\Users\NetPunk\Desktop\dataset\sgns.weibo.char --epochs 0,40
import re
import os
import mindspore
import argparse
from tqdm import tqdm
import mindspore.dataset as ms_dataset
import numpy as np
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
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
    parser.add_argument('--pretrain_url', type=str, default='./resnet/resnet_01.ckpt', help='the pretrain model path')
    parser.add_argument('--rw_path', type=str, default='C:\\datasets\\aclImdb_v1.tar.gz', help='review dataset path')
    parser.add_argument('--weibo_path', type=str, default='C:\\datasets\\glove.6B.zip', help='weibo embedding path')
    parser.add_argument('--output_path', default='save_model/', type=str, help='the path model saved')
    parser.add_argument('--epochs', default='0,10', type=str, help='training epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    # 解析参数
    args_opt = parser.parse_args()
    return args_opt


def replace_all_blank(value):
    # \W 表示匹配非数字字母下划线
    result = re.sub('\W+', '', value).replace("_", '')
    return result


class ReviewData():
    """Review数据集加载器

    加载Review数据集并处理为一个Python迭代对象。

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


def load_rw(rw_path):
    rw_train = ms_dataset.GeneratorDataset(ReviewData(rw_path, "train"), column_names=["text", "label"], shuffle=True)
    rw_test = ms_dataset.GeneratorDataset(ReviewData(rw_path, "test"), column_names=["text", "label"], shuffle=False)
    return rw_train, rw_test


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


def average_last_n_epoch(results_path, n, benchmark):
    epoch = int(args_opt.epochs.split(',')[1])
    match_idxes = [i for i in range(epoch - n, epoch)]
    match_files = []
    files = os.listdir(results_path)
    for file in files:
        if not os.path.isdir(file):
            for idx in match_idxes:
                if re.match('result_ep{}.*'.format(idx), file) is not None:
                    match_files.append(os.path.join(os.getcwd(), 'result', file))
    results = []
    for file in match_files:
        with open(file, mode='r', encoding='utf-8') as f:
            results.append(list(f.read().replace('\n', '')))
    averages = []
    for idx, _ in enumerate(results[0]):
        sum = 0
        for i in range(5):
            sum += int(results[i][idx])
        averages.append(round(sum / n))
    with open(os.path.join(os.getcwd(), 'result', 'average_result_{}.txt'.format(n)), mode='a', encoding='utf-8') as f:
        for ele in averages:
            f.write(str(ele) + '\n')


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


def train_one_epoch(model, train_dataset, epoch=0):
    model.set_train()
    total = train_dataset.get_dataset_size()
    loss_total = 0
    step_total = 0
    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for i in train_dataset.create_tuple_iterator():
            loss = model(*i)
            loss_total += loss.asnumpy()
            step_total += 1
            t.set_postfix(loss=loss_total / step_total)
            t.update(1)


def binary_accuracy(preds, y):
    """
    计算每个batch的准确率
    """

    # 对预测值进行四舍五入
    rounded_preds = np.around(preds)
    correct = (rounded_preds == y).astype(np.float32)
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, valid_dataset, criterion, epoch=0):
    total = valid_dataset.get_dataset_size()
    epoch_loss = 0
    epoch_acc = 0
    step_total = 0
    model.set_train(False)

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for i in valid_dataset.create_tuple_iterator():
            predictions = model(i[0])
            loss = criterion(predictions, i[1])
            epoch_loss += loss.asnumpy()

            acc = binary_accuracy(predictions.asnumpy(), i[1].asnumpy())
            epoch_acc += acc

            step_total += 1
            t.set_postfix(loss=epoch_loss / step_total, acc=epoch_acc / step_total)
            t.update(1)

    return epoch_loss / total, epoch_acc / step_total


def data_preprocessing(vocab, rw_train, rw_test):
    lookup_op = ms_dataset.text.Lookup(vocab, unknown_token='<unk>')
    pad_op = ms_dataset.transforms.c_transforms.PadEnd([500], pad_value=vocab.tokens_to_ids('<pad>'))
    type_cast_op = ms_dataset.transforms.c_transforms.TypeCast(mindspore.float32)

    rw_train = rw_train.map(operations=[lookup_op, pad_op], input_columns=['text'])
    rw_train = rw_train.map(operations=[type_cast_op], input_columns=['label'])

    rw_test = rw_test.map(operations=[lookup_op, pad_op], input_columns=['text'])
    rw_test = rw_test.map(operations=[type_cast_op], input_columns=['label'])

    rw_train, rw_valid = rw_train.split([0.7, 0.3])

    rw_train = rw_train.batch(64, drop_remainder=True)
    rw_valid = rw_valid.batch(64, drop_remainder=True)
    rw_test = rw_test.batch(64, drop_remainder=False)

    return rw_train, rw_test, rw_valid


def predict_sentiment(model, test_dataset, epoch, loss, acc):
    total = test_dataset.get_dataset_size()
    model.set_train(False)

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        path = os.path.join(os.getcwd(), 'result', 'result_ep{}_loss{}_acc{}.txt'.format(epoch, loss, acc))
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


def training_rw(net, loss, rw_train, rw_valid, rw_test, ckpt_file_name, lr):
    net_with_loss = nn.WithLossCell(net, loss)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    train_one_step = nn.TrainOneStepCell(net_with_loss, optimizer)
    best_valid_loss = float('inf')
    epoch_star, epoch_end = args_opt.epochs.split(',')
    # load model
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(net, param_dict)

    for epoch in range(int(epoch_star), int(epoch_end)):
        train_one_epoch(train_one_step, rw_train, epoch)
        valid_loss, valid_acc = evaluate(net, rw_valid, loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(net, ckpt_file_name)

        # generate test results
        predict_sentiment(net, rw_test, epoch, best_valid_loss, valid_acc)


def traning_process(args_opt):
    # load data
    rw_train, rw_test = load_rw(args_opt.rw_path)
    vocab, embeddings = load_weibo(args_opt.weibo_path)

    # check the embedding of word "the"
    idx = vocab.tokens_to_ids('the')
    embedding = embeddings[idx]
    print('the: ', embedding)

    # data preprocessing, adjust tensor shape
    rw_train, rw_test, rw_valid = data_preprocessing(vocab, rw_train, rw_test)

    # setting some hyper parameters
    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = vocab.tokens_to_ids('<pad>')
    ckpt_file_name = os.path.join(args_opt.output_path, 'sentiment-analysis.ckpt')
    loss = nn.BCELoss(reduction='mean')

    # construct model
    net = RNN(embeddings, hidden_size, output_size, num_layers, bidirectional, dropout, pad_idx)

    # use rw for training and save model
    training_rw(net, loss, rw_train, rw_valid, rw_test, ckpt_file_name, args_opt.lr)

    # results_path = os.path.join(os.getcwd(), 'result')
    # n = 5
    # benchmark = int(args_opt.epochs.split(',')[1]) - 1
    # average_last_n_epoch(results_path, n, benchmark)


if __name__ == '__main__':
    args_opt = parse_args()
    traning_process(args_opt)
