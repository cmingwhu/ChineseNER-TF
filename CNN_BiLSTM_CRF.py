"""
 @Author : cmingwhu
 @Datetime : 2019/06/06
 @File : BiLSTM_CRF.py
 @Contact : fccchengm@{zzu.edu.cn}
"""

import os
import sys
import time
from sklearn import metrics

import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.rnn import LSTMCell

from dataUtils import pad_sequences, pad_sequences_, batch_yield_
from eval import conlleval
from utils import get_logger


class CNN_BiLSTM_CRF (object):
    def __init__(self, args, embeddings, char_embedding, tag2label, word2id, chars2ids, paths, config):
        '''
        一些参数的定义
            :param args:
            :param embeddings:
            :param tag2label:
            :param vocab:
            :param paths:
            :param config:
        '''

        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim
        self.embeddings = embeddings
        # self.sequence_lengths = max_sentence
        self.char_vocab = chars2ids
        self.char_dim = args.char_dim
        self.c_embedding = char_embedding
        self.words_lengths = args.max_word
        self.filter_num = args.filter_num
        self.filter_size = args.filter_sizes
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = word2id
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        """
        //构建图
        :return:
        """
        self.add_placeholders()
        self.lookup_layer_op()
        self.CNN_layer_op()
        self.biLSTM_layer_op()
        self.linear_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()


    def add_placeholders(self):
        """
        //对于输入设置占位符，因为真实的数据还没有输入到里面。
        :return:
        """
        # the word id of input sentences
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="words_ids")
        self.char_ids = tf.placeholder(tf.int32, shape=[None,None,None], name="chars_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.update_embedding,
                                           name="_word_embeddings")
            self.word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        # self.word_embeddings = tf.nn.dropout(word_embeddings, keep_prob = (1.0 - self.dropout_pl))

        with tf.variable_scope("chars"):
            _char_embeddings = tf.Variable(self.c_embedding, dtype=tf.float32, trainable=self.update_embedding,
                                           name="_char_embeddings")
            self.char_embeddings = tf.nn.embedding_lookup(params=_char_embeddings,
                                                     ids=self.char_ids,
                                                     name="char_embeddings")
        # self.char_embeddings = tf.nn.dropout(char_embeddings, keep_prob=(1.0 - self.dropout_pl))




    def CNN_layer_op(self):
        """
         CNN for Character-level Representation
         A dropout layer (Srivastava et al., 2014) is applied before character embeddings are input to CNN.
         CNN window size 3
         number of filters 30
        :return:
        """
        with tf.variable_scope('convolution'):
            s = tf.shape(self.char_embeddings)
            cnn_input = tf.reshape(self.char_embeddings, [-1, self.words_lengths, self.char_dim, 1])
            # dropout_applied
            cnn_input = tf.nn.dropout(cnn_input, keep_prob=self.dropout_pl)

            W = tf.get_variable(name='W',
                                shape=[self.filter_size, self.char_dim, 1, self.filter_num],
                                initializer=tf.truncated_normal_initializer(stddev = 0.01),
                                dtype=tf.float32)
            b = tf.get_variable(name='b',
                                shape=[self.filter_num],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            # input dim and filter dim must match
            conv = tf.nn.relu(tf.nn.conv2d(cnn_input, W, strides=[1, 1, 1, 1], padding='VALID') + b)
            char_represent = tf.nn.max_pool(conv, ksize=[1, self.words_lengths-2, 1, 1],
                                            strides=[1,1,1,1], padding='VALID')
            cnn_outputs = tf.reshape(char_represent, [-1, s[1], self.filter_num])

            word_embdding_reshape = tf.reshape(self.word_embeddings, [-1, s[1], self.embedding_dim])

            # character-level representation vector is concatenated
            # with the word embedding vector to feed into
            # the BLSTM network.
            self.concat_operation = tf.concat([cnn_outputs, word_embdding_reshape], 2)

            # return self.concat_operation


    def biLSTM_layer_op(self):
        """
        BiLSTM部分主要用于，根据一个单词的上下文，给出当前单词对应标签的概率分布，可以把BiLSTM看成一个编码层。
        bi-lstm模型搭建
        :return:
        """
        lstm_input = tf.nn.dropout(self.concat_operation, keep_prob=self.dropout_keep_prob)
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=lstm_input,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            self.lstm_outputs = tf.concat([output_fw_seq, output_bw_seq], axis=2)
            #


    def linear_op(self):
        """
        fully contacted beteewn output from bilstm and labels
        :param x:
        :param label_num:
        :return:
        """

        with tf.variable_scope('linear'):
            s = tf.shape(self.lstm_outputs)
            self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1, self.hidden_dim * 2])
            self.lstm_outputs = tf.nn.dropout(self.lstm_outputs, self.dropout_pl)

            W = tf.get_variable(name='W',
                                shape=[2*self.hidden_dim, self.num_tags],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev = 0.01))
            b = tf.get_variable(name='b',
                                shape=[self.num_tags],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            pred = tf.matmul(self.lstm_outputs, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        """
        损失函数定义
        :return:
        """
        if self.CRF:
            # 形状为[num_tags, num_tags]的转移矩阵
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.seqence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)


    def trainstep_op(self):
        """
        //优化器设置
        :return:
        """
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v]
                                   for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def get_feed_dict(self, seqs, words, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :return:
        """

        word_ids, char_ids, seq_len_list = pad_sequences_(seqs, words,self.words_lengths, pad_mark=0)

        feed_dict = {self.word_ids: word_ids, self.sequence_lengths: seq_len_list, self.char_ids: char_ids}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list


    def evaluate__(self, num, predict_labels, labels, sequence_lengths):
        reals = []
        predicts = []
        for i in range(num):
            sentence_len = sequence_lengths[i]
            reals.extend(labels[i][:sentence_len])
            predicts.extend(predict_labels[i])
        p = metrics.precision_score(reals, predicts, average='micro')
        return p

    def predict_one_batch(self, sess, seqs, words):
        """

        :param sess:
        :param seqs: sentence
        :return: predicted labels_list, list of sentences length
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, words, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                # 通俗一点,作用就是返回最好的标签序列.这个函数只能够在测试时使用,在tensorflow外部解码
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list
        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list


    def evaluate(self, label_list, seq_len_list,  data, epoch=None):
        """
        evaluating the results
        :param label_list: predicted labels_list,
        :param seq_len_list: list of sentences length
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch + 1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)


    def dev_one_epoch(self,sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, words, labels in batch_yield_(dev, self.batch_size, self.vocab, self.char_vocab, self.words_lengths,
                                        self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs, words)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def testConll2003(self,sess, dev):
        ids2labels = {0: "O",
                      1: "B-PER", 2: "I-PER",
                      3: "B-LOC", 4: "I-LOC",
                      5: "B-ORG", 6: "I-ORG",
                      7: "B-MISC", 8: "I-MISC",
                      9: "NUL"
                      }

        test_label = []
        predict_label = []
        for seqs, words, labels in batch_yield_(dev, self.batch_size, self.vocab, self.char_vocab, self.words_lengths,
                                                self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs, words)
            label_list, seq_len_list = [], []
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)

            for j in range(len(seq_len_list)):
                sentence_len = seq_len_list[j]
                test_label.extend(labels[j][:sentence_len])
                predict_label.extend(label_list[j])

        target_names = []
        #       remove the padding labels we add
        for i, label in enumerate(test_label):
            test_label[i] = label - 1
        for i, label in enumerate(predict_label):
            predict_label[i] = label - 1
        for i in range(self.num_tags):
            if i == 0:
                continue
            target_names.append(ids2labels[i])
        print(metrics.classification_report(test_label, predict_label, target_names=target_names))

    def run_one_epoch(self,sess, train, dev, tag2label, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size-1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield_(train, self.batch_size, self.vocab, self.char_vocab, self.words_lengths, self.tag2label,
                               shuffle=self.shuffle)
        for step, (seqs, words, labels) in enumerate(batches):
            # sys.stdout.write(" processing: {} batch / {} batches.".format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, words, labels, self.lr, self.dropout_keep_prob)

            # 在运行sess.run函数时，要在代码中明确其需要获取的两个值：[train_op, loss, merged, global_step]
            # 因为要获取这两个值，sess.run() 会返回一个有两个元素的元组。
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)


            label_list_, _  = self.predict_one_batch(sess, seqs, words)
            _, _, lensss = pad_sequences_(seqs, words,self.words_lengths, pad_mark=0)
            p = self.evaluate__(len(lensss), label_list_, labels, lensss)


            if step + 1 == 1 or (step + 1) % 10 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}, P:{:.4}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num,p))
            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)
                # label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
                # self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)


    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=================== testing ====================')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)


