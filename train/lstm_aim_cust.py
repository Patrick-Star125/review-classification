import re
import math
import argparse
import string
import os
import tarfile

import six
import numpy as np
import mindspore
import mindspore.dataset as ms_dataset
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops

from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore import set_context, PYNATIVE_MODE
from mindspore import Tensor
from mindspore.common.initializer import Uniform, HeUniform
from mindspore.train.callback import Callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from tqdm import tqdm
from aim import Run


set_context(mode=PYNATIVE_MODE)


def parse_args():
    """

    parse argparse
    :return: args
    """
    parser = argparse.ArgumentParser(description="train lstm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pretrain_path',
                        type=str,
                        default='./resnet/resnet_01.ckpt',
                        help='the pretrain model path')

    parser.add_argument('--imdb_path',
                        type=str,
                        default='datasets/drizzlezyk/imdb/aclImdb_v1.tar.gz',
                        help='imbd path')

    parser.add_argument('--glove_path',
                        type=str,
                        default='datasets/drizzlezyk/imdb/glove.6B/',
                        help='glove path')

    parser.add_argument('--output_path',
                        default='train/save_model/',
                        type=str,
                        help='the path model saved')

    parser.add_argument('--aim_repo',
                        default='./db',
                        type=str,
                        help='the path aim file saved')

    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='training epochs')

    parser.add_argument('--lr',
                        default=0.0001,
                        type=float,
                        help='learning rate')

    return parser.parse_args()


class AimCallback(Callback):
    def __init__(self, model, dataset_test, aim_run):
        super(AimCallback, self).__init__()
        self.aim_run = aim_run  # 传入aim实例
        self.model = model  # 传入model，用于eval
        self.dataset_test = dataset_test  # 传入dataset_test, 用于eval test

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        # loss
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        loss = cb_params.net_outputs
        self.aim_run.track(float(str(loss)), name='loss', step=step_num, epoch=epoch_num,
                                  context={"subset": "train"})

    def epoch_end(self, run_context):
        """epoch end"""
        cb_params = run_context.original_args()
        # loss
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        loss = cb_params.net_outputs
        train_dataset = cb_params.train_dataset
        train_acc = self.model.eval(train_dataset)
        test_acc = self.model.eval(self.dataset_test)
        print("【Epoch:】", epoch_num, "【Step:】", step_num, "【loss:】", loss,
              "【train_acc:】", train_acc['accuracy'], "【test_acc:】",
              test_acc['accuracy'])

        self.aim_run.track(float(str(loss)),
                           name='loss',
                           epoch=epoch_num,
                           context={"subset": "train"})

        self.aim_run.track(float(str(train_acc['accuracy'])),
                           name='accuracy',
                           epoch=epoch_num,
                           context={"subset": "train"})

        self.aim_run.track(float(str(test_acc['accuracy'])),
                           name='test_accuracy',
                           epoch=epoch_num,
                           context={"subset": "test"})


class IMDBData():
    """IMDB数据集加载器

    加载IMDB数据集并处理为一个Python迭代对象。

    """
    label_map = {
        "pos": 1,
        "neg": 0
    }

    def __init__(self, path, mode="train"):
        self.mode = mode
        self.path = path
        self.docs, self.labels = [], []

        self._load("pos")
        self._load("neg")

    def _load(self, label):
        pattern = re.compile(r"aclImdb/{}/{}/.*\.txt$".format(self.mode, label))
        # 将数据加载至内存
        with tarfile.open(self.path) as tarf:
            temp_tarf = tarf.next()
            while temp_tarf is not None:
                if bool(pattern.match(temp_tarf.name)):
                    # 对文本进行分词、去除标点和特殊字符、小写处理
                    self.docs.append(str(tarf.extractfile(temp_tarf).read().rstrip(six.b("\n\r"))
                                         .translate(None, six.b(string.punctuation)).lower()).split())
                    self.labels.append([self.label_map[label]])
                temp_tarf = tarf.next()

    def __getitem__(self, idx):
        return self.docs[idx], self.labels[idx]

    def __len__(self):
        return len(self.docs)


def load_imdb(imdb_path):
    imdb_train = ms_dataset.GeneratorDataset(IMDBData(imdb_path, "train"),
                                             column_names=["text", "label"],
                                             shuffle=True)
    imdb_test = ms_dataset.GeneratorDataset(IMDBData(imdb_path, "test"),
                                            column_names=["text", "label"],
                                            shuffle=False)
    return imdb_train, imdb_test


def load_glove(glove_path):
    glove_100d_path = os.path.join(glove_path, 'glove.6B.100d.txt')
    embeddings = []
    tokens = []
    with open(glove_100d_path, encoding='utf-8') as gf:
        for glove in gf:
            word, embedding = glove.split(maxsplit=1)
            tokens.append(word)
            embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))

    # 添加 <unk>, <pad> 两个特殊占位符对应的embedding
    embeddings.append(np.random.rand(100))
    embeddings.append(np.zeros((100,), np.float32))

    vocab = ms_dataset.text.Vocab.from_list(tokens,
                                            special_tokens=["<unk>", "<pad>"],
                                            special_first=False)
    embeddings = np.array(embeddings).astype(np.float32)
    return vocab, embeddings


class RNN(nn.Cell):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      embedding_table=Tensor(embeddings),
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
            t.set_postfix(loss=loss_total/step_total)
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


def evaluate(model, test_dataset, criterion, epoch=0):
    total = test_dataset.get_dataset_size()
    epoch_loss = 0
    epoch_acc = 0
    step_total = 0
    model.set_train(False)

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for i in test_dataset.create_tuple_iterator():
            predictions = model(i[0])
            loss = criterion(predictions, i[1])
            epoch_loss += loss.asnumpy()

            acc = binary_accuracy(predictions.asnumpy(), i[1].asnumpy())
            epoch_acc += acc

            step_total += 1
            t.set_postfix(loss=epoch_loss/step_total, acc=epoch_acc/step_total)
            t.update(1)

    return epoch_loss / total


def data_preprocessing(vocab, imdb_train, imdb_test, batch_size):
    lookup_op = ms_dataset.text.Lookup(vocab, unknown_token='<unk>')
    pad_op = ms_dataset.transforms.c_transforms.PadEnd([500], pad_value=vocab.tokens_to_ids('<pad>'))
    type_cast_op = ms_dataset.transforms.c_transforms.TypeCast(mindspore.float32)

    imdb_train = imdb_train.map(operations=[lookup_op, pad_op], input_columns=['text'])
    imdb_train = imdb_train.map(operations=[type_cast_op], input_columns=['label'])

    imdb_test = imdb_test.map(operations=[lookup_op, pad_op], input_columns=['text'])
    imdb_test = imdb_test.map(operations=[type_cast_op], input_columns=['label'])

    imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])

    imdb_train = imdb_train.batch(batch_size, drop_remainder=True)
    imdb_valid = imdb_valid.batch(batch_size, drop_remainder=True)

    return imdb_train, imdb_test, imdb_valid


def training_imdb(net, loss, imdb_train, imdb_valid, ckpt_file_name, lr, aim_run):
    net_with_loss = nn.WithLossCell(net, loss)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    train_one_step = nn.TrainOneStepCell(net_with_loss, optimizer)
    best_valid_loss = float('inf')

    for epoch in range(args_opt.epochs):
        train_one_epoch(train_one_step, imdb_train, epoch)
        valid_loss = evaluate(net, imdb_valid, loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(net, ckpt_file_name)

        aim_run.track(float(str(valid_loss)), name='loss', epoch=epoch,
                      context={"subset": "validate"})


def predict_sentiment(model, vocab, sentence, score_map):
    model.set_train(False)
    tokenized = sentence.lower().split()
    indexed = vocab.tokens_to_ids(tokenized)
    tensor = mindspore.Tensor(indexed, mindspore.int32)
    tensor = tensor.expand_dims(0)
    prediction = model(tensor)
    return score_map[int(np.round(prediction.asnumpy()))]


def traning_process(args_opt):
    # load data
    imdb_train, imdb_test = load_imdb(args_opt.imdb_path)
    vocab, embeddings = load_glove(args_opt.glove_path)

    # check the embedding of word "the"
    idx = vocab.tokens_to_ids('the')
    embedding = embeddings[idx]
    print('the: ', embedding)

    # setting some hyper parameters
    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = vocab.tokens_to_ids('<pad>')
    ckpt_file_name = os.path.join(args_opt.output_path, '../sentiment-analysis.ckpt')
    loss = nn.BCELoss(reduction='mean')

    batch_size_choice = [64, 128]
    learning_rate_choice = [0.0001, 0.001]

    for batch_size in batch_size_choice:
        for learning_rate in learning_rate_choice:
            aim_run = Run(repo=args_opt.aim_repo,
                          experiment=f"{args_opt.output_path}/bs{batch_size}_lr{learning_rate}")
            aim_run['learning_rate'] = learning_rate
            aim_run['batch_size'] = batch_size

            imdb_train, imdb_test, imdb_valid = data_preprocessing(vocab,
                                                                   imdb_train,
                                                                   imdb_test,
                                                                   batch_size)
            net = RNN(embeddings,
                      hidden_size,
                      output_size,
                      num_layers,
                      bidirectional,
                      dropout,
                      pad_idx)

            # use imdb for training and save model
            training_imdb(net, loss, imdb_train, imdb_valid, ckpt_file_name, learning_rate, aim_run)


if __name__ == '__main__':
    args_opt = parse_args()
    traning_process(args_opt)
