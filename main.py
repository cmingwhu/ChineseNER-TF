"""
 @Author : cmingwhu
 @Datetime : 2019/06/06
 @File : BiLSTM_CRF.py
 @Contact : fccchengm@{zzu.edu.cn}
"""

import tensorflow as tf
import numpy as np
import os, argparse, time, random
from BiLSTM_CRF import BiLSTM_CRF
from CNN_BiLSTM_CRF import CNN_BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from dataUtils import read_corpus, read_dictionary, tag2label_mapping, random_embedding, vocab_build, \
    build_pretrained_words_embeddings, char_embedding_matrix,read_sentences,char2id


# Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # need ~700MB GPU memory


# hyper parameters
parser = argparse.ArgumentParser(description='Chinese/English NER task')
parser.add_argument('--dataset_name', type=str, default='Conll2003_NER',
                    help='choose a dataset(Conll2003_NER,MSRA, ResumeNER, WeiboNER,人民日报)')
parser.add_argument('--CE_Flag', type=str, default='0', help='Choose languages, 0:E,1:C!')
parser.add_argument('--use_char', type=str, default='1', help='use char')
# parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
# parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=10, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=50, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=200, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--filter_num', type=int, default=30, help='number of convolution kenerls')
parser.add_argument('--filter_sizes', type=int, default=3, help='Comma-separated filter sizes')
parser.add_argument('--char_dim', type=int, default=30, help='char embedding dimension')
parser.add_argument('--max_word', type=int, default=20, help='the max length of word in dataset')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--use_pre_emb', type=str2bool, default=False,
                    help='use pre_trained char embedding or init it randomly')
parser.add_argument('--pretrained_emb_path', type=str, default='./sgns.wiki.char', help='pretrained embedding path')
parser.add_argument('--embedding_dim', type=int, default=100, help='random init word embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=False, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1584154467', help='model for test and demo')
args = parser.parse_args()



# read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('data_path', args.dataset_name, 'train.txt')
    test_path = os.path.join('data_path', args.dataset_name, 'test.txt')
    # dev_path = os.path.join('data_path', args.dataset_name, 'valid.txt')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    # dev_data = read_corpus(dev_path)

    # English CNN_BiLSTM_CRF
    if args.CE_Flag == '0':
        train_sen, _ = read_sentences(train_path)
        test_sen, _ = read_sentences(test_path)

        # initializing char embedding
        chars2ids, char_vocabulary, char_embedding = char_embedding_matrix(train_sen + test_sen, args.char_dim)



# vocabulary build
if not os.path.exists(os.path.join('data_path', args.dataset_name, 'word2id.pkl')):
    vocab_build(os.path.join('data_path', args.dataset_name, 'word2id.pkl'),
                os.path.join('data_path', args.dataset_name, 'train.txt'),os.path.join('data_path', args.dataset_name, 'test.txt'))

# train_chars2ids  = char2id(chars2ids, char_vocabulary, train_sen, max_sentence, max_word)
# test_chars2ids = char2id(chars2ids, char_vocabulary, test_sen, max_sentence, max_word)

# get word dictionary
word2id = read_dictionary(os.path.join('data_path', args.dataset_name, 'word2id.pkl'))

# build words embeddings
if not args.use_pre_emb:
    embeddings = random_embedding(word2id, args.embedding_dim)
    log_pre = 'not_use_pretrained_embeddings'
else:
    pre_emb_path = os.path.join('.', args.pretrained_emb_path)
    embeddings_path = os.path.join('data_path', args.dataset_name, 'pretrain_embedding.npy')
    if not os.path.exists(embeddings_path):
        build_pretrained_words_embeddings(pre_emb_path, embeddings_path, word2id, args.embedding_dim)
    embeddings = np.array(np.load(embeddings_path), dtype='float32')
    log_pre = 'use_pretrained_embeddings'

# choose tag2label
tag2label = tag2label_mapping[args.dataset_name]




# paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('model_path', args.dataset_name, timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, args.dataset_name + log_pre + "_log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

# training model
if args.mode == 'train':
    # train model on the whole training data
    print("train data: {}".format(len(train_data)))
    print("test data: {}".format(len(test_data)))
    # print("dev data: {}".format(len(dev_data)))

    if args.use_char == '0':
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        model.build_graph()

        model.train(train=train_data, dev=test_data)  # use test_data.txt as the dev_data to see overfitting phenomena
    elif args.use_char == '1':
        model = CNN_BiLSTM_CRF(args, embeddings, char_embedding, tag2label, word2id, chars2ids, paths, config=config)
        model.build_graph()

        model.train(train=train_data, dev=test_data)  # use test_data.txt as the dev_data to see overfitting phenomena



# testing model
elif args.mode == 'test':
    if args.use_char == '0':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        model.build_graph()
        print("test data: {}".format(len(test_data)))
        model.test(test_data)
    elif args.use_char == '1':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file
        model = CNN_BiLSTM_CRF(args, embeddings, char_embedding, tag2label, word2id, chars2ids, paths, config=config)
        model.build_graph()
        print("test data: {}".format(len(test_data)))
        model.test(test_data)

# demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while (1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))