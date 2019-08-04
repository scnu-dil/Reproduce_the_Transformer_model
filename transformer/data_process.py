# 数据组织类

#-*-coding:utf-8-*-

import re
import unicodedata

import torch
from torch.autograd import Variable

import os

import sys
sys.path.append(os.path.abspath('..'))
from parameters import Parameters
param = Parameters()


# 字符统计类
class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        # self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.index2word = {0: '<blank>', 1: '<unk>', 2: '<s>', 3: '</s>'}
        self.n_words = 4  # 初始字典有4个字符
        self.seq_max_len = 0

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('保留单词数 %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        # self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.index2word = {0: '<blank>', 1: '<unk>', 2: '<s>', 3: '</s>'}
        self.n_words = 4  # Count default tokens

        for word in keep_words:
            self.index_word(word)


# 数据准备类
class DataProcess(object):

    def __init__(self, train_src, train_tgt, val_src, val_tgt, test_src, test_tgt):
        self.train_src = train_src
        self.train_tgt = train_tgt

        self.val_src = val_src
        self.val_tgt = val_tgt

        self.test_src = test_src
        self.test_tgt = test_tgt

        self.src_max_len = 2  # 序列前后有SOS和EOS
        self.tgt_max_len = 2  # 序列前后有SOS和EOS

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([,.!?])", r" \1 ", s)
        s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def filter_pairs(self, pairs):
        filtered_pairs = []
        for pair in pairs:
            sentence_num = 0
            for i in pair:
                if len(i.split(' ')) > param.min_len and len(i.split(' ')) <= param.max_len:
                    sentence_num += 1
            if sentence_num == len(pair):
                filtered_pairs.append(pair)
            else:
                temp = [len(i.split(' ')) for i in pair]
                print(temp, pair)

        return filtered_pairs

    def indexes_from_sentence(self, lang, sentence):
        # 前后加上sos和eos。注意句子的句号也要加上，如果这个词没有出现在词典中（已经去除次数小于限定的词），以unk填充
        return [param.sos] + [lang.word2index.get(word, param.unk) for word in sentence.split(' ')] + [param.eos]

    def read_file(self, data):
        content = open(data, encoding='utf-8').read().split('\n')  # 读取文件并处理
        content = self.filter_pairs([self.normalize_string(s) for s in content])  # 规范化字符，并限制长度

        return content

    # 返回
    def get_src_tgt_data(self):
        src_content = []
        for i in (self.train_src, self.val_src, self.test_src):
            src_content += self.read_file(i)
        src_lang = Lang('src')  # source字符类

        tgt_content = []
        for j in (self.train_tgt, self.val_tgt, self.test_tgt):
            tgt_content += self.read_file(j)
        tgt_lang = Lang('tgt')  # target字符类

        src_tgt = []  # 存储source和target序列
        for line in range(len(src_content)):  # 检索单词
            src_lang.index_words(src_content[line])
            tgt_lang.index_words(tgt_content[line])
            src_tgt.append([src_content[line], tgt_content[line]])

        # 修剪单词表，少于限定次数将被删除
        src_lang.trim(param.min_word_count)
        tgt_lang.trim(param.min_word_count)

        self.src_max_len += max([len(s[0].split(' ')) for s in src_tgt])  # 全部输入序列的最大长度
        self.tgt_max_len += max([len(s[1].split(' ')) for s in src_tgt])  # 全部目标序列的最大长度

        return src_tgt, src_lang, tgt_lang

    def word_2_index(self, mode, src_lang, tgt_lang):
        src = 0
        tgt = 0
        if mode == 'train':
            src = self.train_src
            tgt = self.train_tgt
        elif mode == 'val':
            src = self.val_src
            tgt = self.val_tgt
        elif mode == 'test':
            src = self.test_src
            tgt = self.test_tgt

        src_seq = self.read_file(src)
        tgt_seq = self.read_file(tgt)

        # 判断输入和目标序列的句子数量是否对应，这里或许可以使用assert断言
        if len(src_seq) != len(tgt_seq):
            print('输入与目标句子不对应！！！')
            exit()

        # 以list存储批序列
        src_list = []
        tgt_list = []
        src_tgt_list = []

        for i in range(len(src_seq)):  # 序列字符token转化为索引token
            src_list.append(self.indexes_from_sentence(src_lang, src_seq[i]))
            tgt_list.append(self.indexes_from_sentence(tgt_lang, tgt_seq[i]))
            src_tgt_list.append([str(self.indexes_from_sentence(src_lang, src_seq[i])),
                                 str(self.indexes_from_sentence(tgt_lang, tgt_seq[i]))])

        return src_list, tgt_list, src_tgt_list
