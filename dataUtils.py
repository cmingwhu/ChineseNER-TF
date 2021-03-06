"""
 @Author : cmingwhu
 @Datetime : 2019/06/06
 @File : BiLSTM_CRF.py
 @Contact : fccchengm@{zzu.edu.cn}
"""
import os
import pickle
import codecs
import numpy as np
import random


tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }
# Conll2003_ner tags, BIO
tag2label_Conll2003 = {"O": 0,
                       "B-PER": 1, "I-PER": 2,
                       "B-LOC": 3, "I-LOC": 4,
                       "B-ORG": 5, "I-ORG": 6,
                       "B-MISC": 7, "I-MISC": 8,
                       "NUL": 9
                       }

# CCKS2017 tags, BIO
tag2label_CCKS17 = {"O": 0,
                  "B-SIGNS": 1, "I-SIGNS": 2,
                  "B-CHECK": 3, "I-CHECK": 4,
                  "B-DISEASE": 5, "I-DISEASE": 6,
                  "B-TREATMENT": 7, "I-TREATMENT": 8,
                  "B-BODY": 9, "I-BODY": 10
                    }

# CCKS2018 tags, BIO
tag2label_CCKS18 = {"O": 0,
                  "B-ANATOMICSITE": 1, "I-ANATOMICSITE": 2,
                  "B-SYMPTOMDES": 3, "I-SYMPTOMDES": 4,
                  "B-INDEPSYMPTOM": 5, "I-INDEPSYMPTOM": 6,
                  "B-DRUGS": 7, "I-DRUGS": 8,
                  "B-OPERATION": 9, "I-OPERATION": 10
                    }

# 默认数据集 MSRA tags, BIO
tag2label_msra = {"O": 0,
                  "B-PER": 1, "I-PER": 2,
                  "B-LOC": 3, "I-LOC": 4,
                  "B-ORG": 5, "I-ORG": 6
                  }

# 人民日报数据集
tag2label_chinadaily = {"O": 0,
                        "B-PERSON": 1, "I-PERSON": 2,
                        "B-LOC": 3, "I-LOC": 4,
                        "B-ORG": 5, "I-ORG": 6,
                        "B-GPE": 7, "I-GPE": 8,
                        "B-MISC": 9, "I-MISC": 10
                        }
# WeiboNER
tag2label_weibo_ner = {"O": 0,
                       "B-PER.NAM": 1, "I-PER.NAM": 2,
                       "B-LOC.NAM": 3, "I-LOC.NAM": 4,
                       "B-ORG.NAM": 5, "I-ORG.NAM": 6,
                       "B-GPE.NAM": 7, "I-GPE.NAM": 8,
                       "B-PER.NOM": 9, "I-PER.NOM": 10,
                       "B-LOC.NOM": 11, "I-LOC.NOM": 12,
                       "B-ORG.NOM": 13, "I-ORG.NOM": 14
                       }

# Resume_NER
tag2label_resume_ner = {"O": 0,
                        "B-NAME": 1, "M-NAME": 2, "E-NAME": 3, "S-NAME": 4,
                        "B-RACE": 5, "M-RACE": 6, "E-RACE": 7, "S-RACE": 8,
                        "B-CONT": 9, "M-CONT": 10, "E-CONT": 11, "S-CONT": 12,
                        "B-LOC": 13, "M-LOC": 14, "E-LOC": 15, "S-LOC": 16,
                        "B-PRO": 17, "M-PRO": 18, "E-PRO": 19, "S-PRO": 20,
                        "B-EDU": 21, "M-EDU": 22, "E-EDU": 23, "S-EDU": 24,
                        "B-TITLE": 25, "M-TITLE": 26, "E-TITLE": 27, "S-TITLE": 28,
                        "B-ORG": 29, "M-ORG": 30, "E-ORG": 32, "S-ORG": 33,
                        }

tag2label_mapping = {
    'MSRA': tag2label_msra,
    '人民日报': tag2label_chinadaily,
    'WeiboNER': tag2label_weibo_ner,
    'ResumeNER': tag2label_resume_ner,
    'CCKS17': tag2label_CCKS17,
    'CCKS18': tag2label_CCKS18,
    'Conll2003_NER':tag2label_Conll2003
}


def build_pretrained_words_embeddings(pretrained_emb_path, embeddings_path, word2id, embedding_dim):
    """
    load pretrained words embeddings
    :param pretrained_emb_path:
    :param embeddings_path:
    :param word2id:
    :param embedding_dim:
    :return:
    """
    pre_emb = {}
    for line in codecs.open(pretrained_emb_path, 'r', 'utf-8'):
        line = line.strip().split()
        if len(line) == embedding_dim + 1:
            pre_emb[line[0]] = [float(x) for x in line[1:]]
    word_ids = sorted(word2id.items(), key=lambda x: x[1])
    characters = [c[0] for c in word_ids]
    embeddings = list()
    for i, ch in enumerate(characters):
        if ch in pre_emb:
            embeddings.append(pre_emb[ch])
        else:
            embeddings.append(np.random.uniform(-0.25, 0.25, embedding_dim).tolist())
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(embeddings_path, embeddings)


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data= (char,label)
    eg: 中 B-ORG
        国 O
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def read_sentences(path, encoding="utf8"):
    """
    read sentences and labels from dataset
    :param path:
    :param encoding:
    :return:
    """
    with open(path) as fp:
        sentence = []
        sentences = []
        label = []
        labels = []
        for line in fp.readlines()[2:]:
            if line == '\n':
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
            else:
                sentence.append(line.split()[0])
                label.append(line.split()[-1])
    return (sentences, labels)


def vocab_build(vocab_path, corpus_path,test_path, min_count=1, CE_Flag = '0'):
    """
    Chinese create vocabulary, word to id, word2id={[word, [word_id, word_freq]]}
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    dataa = read_corpus(corpus_path)
    test_data = read_corpus(test_path)
    data = dataa+ test_data
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if CE_Flag is '1':
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                    word = '<ENG>'
                if word not in word2id:
                    word2id[word] = [len(word2id) + 1, 1]
                else:
                    word2id[word][1] += 1
            else:
                if word.isdigit():
                    word = '<NUM>'
                if word not in word2id:
                    word2id[word] = [len(word2id) + 1, 1]
                else:
                    word2id[word][1] += 1
    # delete words for word_freq <1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if CE_Flag is '1':
            if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
                low_freq_words.append(word)
        else:
            if word_freq < min_count and word != '<NUM>':
                low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

def sentence2id(sent, word2id):
    """
    Based on transform sentence to id
    :param sent:
    :param word2id:
    :return: sentence2id
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

def sentence2id_E(sent, word2id):
    """
    Based on transform sentence to id
    :param sent:
    :param word2id:
    :return: sentence2id
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

def read_dictionary(vocab_path):
    """
    read dictionary plk
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id



def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def char_embedding_matrix(sentences, dim):
    """
    randomly initial char embedding
    :param sentences:
    :param d:
    :return:
    """
    chars = []
    chars_embedding = []
    char_id = {}
    for sentence in sentences:
        for word in sentence:
            for char in word:
                chars.append(char)
    chars = set(chars)
    #   add a null symbol for char padding to CNN to represent character-level---------------
    char_id['NUL'] = 0
    #   add a unknown symbol in case the test dataset appear the chars which are not appeared in training dataset
    char_id['<UNK>'] = len(chars) + 1
    chars_embedding.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (dim, 1)))
    for i, char in enumerate(chars):
        char_id[char] = i + 1
        chars_embedding.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (dim, 1)))
    chars_embedding.append(np.random.uniform(-np.sqrt(3.0 / 30), np.sqrt(3.0 / 30), (dim, 1)))
    chars_embedding = np.reshape(chars_embedding, [-1, dim])
    chars_embedding = chars_embedding.astype(np.float32)
    return (char_id, chars, chars_embedding)

def char2id(sent_, chars_vocab, max_word):
    """
    convert chars in batch sentences to ids
    :param char_id:
    :param vocabulary:
    :param X:
    :param max_sentence:
    :param max_word:
    :return:
    """
    X2id = []
    for word in sent_:
        char_index = []
        for char in word:
            if char.isdigit():
                char = '<NUM>'
            if char not in chars_vocab:
                char = '<UNK>'
            char_index.append(chars_vocab[char])
        char_index_ = char_index[:max_word] + [0] * max(max_word - len(char_index), 0)
        X2id.append(char_index_)
    return X2id


def build_label_ids(labels):
    """

    :param labels:
    :return:
    """
    label = []
    label_id = {}
    for sentence in labels:
        for word in sentence:
            label.append(word)
    label = set(label)
    #    label_num = len(label)
    label_id['NUL'] = 0
    for i, label in enumerate(label):
        label_id[label] = i + 1
    id_label = {v: k for k, v in label_id.items()}
    label_num = len(label_id.values())

    return (label_id, id_label, label_num)


def pad_sequences(sequences, pad_mark=0):
    """
    Filling sequences
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def pad_sequences_(sequences, words, max_word, pad_mark=0):
    """
    Filling sequences
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list,word_list, seq_len_list = [], [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    tep = np.zeros(max_word,dtype=int).tolist()
    for cha in words:
        char_ind = []
        for cha1 in cha:
            char_ind.append(cha1)
        for l in range(max_len-len(char_ind)):
           char_ind.append(tep)
        word_list.append(char_ind)

    return seq_list, word_list, seq_len_list



def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def batch_yield_(data, batch_size, vocab, chars_vocab, max_word, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, words, labels = [], [], []
    for (sent_, tag_) in data:
        word_ = char2id(sent_, chars_vocab, max_word)
        sent_ = sentence2id_E(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, words, labels
            seqs, words, labels = [], [], []

        seqs.append(sent_)
        words.append(word_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, words, labels




if __name__ == '__main__':

    if not os.path.exists(os.path.join('data_path', 'CCKS17', 'word2id.pkl')):
        vocab_build(os.path.join('data_path', 'CCKS17', 'word2id.pkl'),
                    os.path.join('data_path', 'CCKS17',  'train_data.txt'))

    word2id = read_dictionary(os.path.join('data_path', 'CCKS17', 'word2id.pkl'))

    build_pretrained_words_embeddings('./data_path/sgns.wiki.char', './data_path/CCKS17/vectors.npy', word2id, 300)
