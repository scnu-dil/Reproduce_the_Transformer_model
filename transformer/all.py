# -*-coding:utf-8-*-

# transformer的具体模块实现

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import os
import math
import datetime
from tqdm import tqdm, trange

import sys
sys.path.append(os.getcwd())
from data_process import DataProcess

sys.path.append(os.path.abspath('..'))
from parameters import Parameters
param = Parameters()

"""
B batch_size
L max_len
D dimension = d_mode
d dimension = d_q, d_k, d_v
H heads
"""

# 一些必须的工具函数类
class Tools():

    def __init__(self):
        pass

    def str_2_list(self, tgt_seq):
        """

        :param tgt_seq:
        :return:
        """
        ss = []
        for s in tgt_seq:  # 把字符序列还原list序列
            s = s.lstrip('[').rstrip(']').replace(' ', '').split(',')
            ss.append([int(i) for i in s])

        return ss

    def pad_seq(self, seq, max_len):
        """

        :param seq:
        :param max_len:
        :return:
        """
        seq += [param.pad for i in range(max_len - len(seq))]
        return seq

    def seq_2_tensor(self, seq):
        """

        :param seq:
        :return:
        """
        seq = self.str_2_list(seq)  # 还原序列
        seq_max_len = max(len(s) for s in seq)  # 该批次序列中的最大长度
        # 以最大长度补齐该批次的序列并转化为tensor
        seq = Variable(torch.LongTensor([self.pad_seq(s, seq_max_len) for s in seq]))

        return seq.to(device=param.device)

    def batch_2_tensor(self, batch_data):
        """

        :param batch_data: B*L 该批次的数据
        :return:
        """
        src_seq = self.seq_2_tensor(batch_data[0])  # 生成并转化source序列
        tgt_seq = self.seq_2_tensor(batch_data[1])  # 生成并转化target序列

        return src_seq, tgt_seq

    def seq_2_pos(self, seq):
        """

        :param seq:
        :return:
        """
        batch_size, seq_max_len = seq.shape
        pos = np.zeros((batch_size, seq_max_len))

        for idx_1, i in enumerate(seq):
            for idx_2, j in enumerate(i):
                if int(j.cpu().detach().numpy()) != 0:
                    pos[idx_1][idx_2] = idx_2 + 1
                else:
                    continue

        return torch.LongTensor(pos).to(device=param.device)


tool = Tools()

# Scaled Dot-Product Attention的实现类
class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制

    An attention function can be described as a query and a set of key-value pairs to an output,
    where the query, keys, values, and output are all vectors.
    The output is computed as a weighted sum of the values,
    where the weight assigned to each value is computed by a compatibility of the query with the corresponding key.
    通过确定Q和K之间的相似程度来选择V

    Attention(Q, K, V)=softmax(Q*K^T/d_k……1/2)*V

    d_k表示K的维度，默认64，当前点积得到的结果维度很大的时候，那么经过softmax函数的作用？梯度是处于很小的区域，
    这样是不利于BP的（也就是梯度消失的问题），除以一个缩放因子，能降低点积结果维度，缓解梯度消失问题

    在encoder的self-attention中，Q, K, V来自于同一个地方（相等），都是上一层encoder的输出。
    第一层的Q, K，V是word embedding和positional encoding相加得到的输入。

    在decoder的self-attention中，Q, K, V来自于同一个地方（相等），都是上一层decoder的输出。
    第一层的Q, K，V是word embedding和positional encoding相加得到的输入。
    但是在decoder中是基于以前信息来预测下一个token的，所以需要屏蔽当前时刻之后的信息，即做sequence masking

    在encoder-decoder交互的context attention层中，Q来自于decoder的上一层的输出，K和V来自于encoder的输出，K和V相同

    Q, K, V三个始终的维度是一样的，d_q=d_k=d_v
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(param.dropout)  # dropout操作
        self.softmax = nn.Softmax(dim=2)  # softmax操作，TODO 为什么是定义在dim=2

    def forward(self, q, k, v, mask):
        """
        # softmax(q*k^T/d_k^(1/2))*v
        :param q: query (B*h)*L*d
        :param k: key (B*h)*L*d
        :param v: value (B*h)*L*d
        :param mask: (B*h)*L*L
        :return:
        """
        attn = torch.bmm(q, k.transpose(1, 2))  # q*k^T (B*h)*L*L

        attn = attn / np.power(param.d_k, 0.5)  # q*k^T / d_k^(1/2)

        attn = attn.masked_fill(mask, -np.inf)  # 屏蔽掉序列补齐的位置 (B*h)*L*L

        attn = self.softmax(attn)

        attn = self.dropout(attn)

        attn = torch.bmm(attn, v)  # *v D*L*d

        return attn


# Position-wise Feed-Forward Networks的实现类
class PositionWiseFeedForward(nn.Module):
    """
    按位置的前馈网络
    包含两个线性变换和一个非线性函数
    FeedForwardNet(x)=max(0, x*W_1+b_1)*W_2+b_2
    采用两个一维核大小=1的一维卷积解释 TODO 这个怎么计算的
    """
    def __init__(self):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(in_channels=param.d_model,
                             out_channels=param.d_ff,
                             kernel_size=1)  # kernel_size=1，权重w_1层

        self.w_2 = nn.Conv1d(in_channels=param.d_ff,
                             out_channels=param.d_model,
                             kernel_size=1)  # kernel_size=1，权重w_2层
        # TODO 对权重层的参数进行初始化
        self.dropout = nn.Dropout(param.dropout)
        self.layer_norm = nn.LayerNorm(param.d_model)  # LN层归一化层

    def forward(self, x):
        residual = x  # 残差 B*L*(h*d=d_model)

        ffn = F.relu(self.w_1(x.transpose(1, 2)))  # max(0, x*w_1+b1) B*d_ff*L
        ffn = self.w_2(ffn)  # *w_2+b_2 B*d_ff*L

        ffn = self.dropout(ffn)
        ffn = self.layer_norm(residual + ffn.transpose(1, 2))  # 残差+LayerNorm结构 B*L*D

        return ffn


# Multi-head Attention的实现类
class MultiHeadAttention(nn.Module):
    """
    multi-head attention多头注意力机制
    将Q，K，V通过线性映射（乘上权重层W），分成h(h=8)份，每一份再所缩放点积操作效果更好
    再把h份合起来，经过线性映射（乘上权重层W），得到最后的输出

    MultiHead(Q, K, V)=Concat(head_1, head_2, ..., head_h) * W^O
        head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)

    每一份做缩放点积操作的d_k, d_q, d_v的维度是原来总维度的h份，
    即d_k，d_q，d_v的维度=D_k/h，D_q/h，D_v/h

    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        # query, key, value的权重层
        self.w_q_s = nn.Linear(param.d_model, param.heads * param.d_q)
        self.w_k_s = nn.Linear(param.d_model, param.heads * param.d_k)
        self.w_v_s = nn.Linear(param.d_model, param.heads * param.d_v)
        # 权重层以正态分布初始化
        nn.init.normal_(self.w_q_s.weight, mean=0, std=np.sqrt(2.0 / (param.d_model + param.d_q)))
        nn.init.normal_(self.w_k_s.weight, mean=0, std=np.sqrt(2.0 / (param.d_model + param.d_k)))
        nn.init.normal_(self.w_v_s.weight, mean=0, std=np.sqrt(2.0 / (param.d_model + param.d_v)))

        self.concat_heads = nn.Linear(param.heads * param.d_v, param.d_model)  # 级联多头

        self.attention = ScaledDotProductAttention()  # 缩放点积attention计算

        self.dropout = nn.Dropout(param.dropout)

        self.layer_norm = nn.LayerNorm(param.d_model)  # LN层归一化操作

    def forward(self, query, key, value, mask):
        """

        :param q: query B*L*D
        :param k: key B*L*D
        :param v: value B*L*D
        :param mask: 屏蔽位 B*L*L
        :return: h个头的缩放点积计算结果 B*L*(h*d=d_model)
        """
        residual = query  # 残差

        batch_size, seq_len_q, dim = query.shape
        batch_size, seq_len_k, dim = key.shape
        batch_size, seq_len_v, dim = value.shape

        # query乘上权重得到一个头（但是同时计算多头） B*L*H*d
        query = self.w_q_s(query).view(batch_size, seq_len_q, param.heads, param.d_q)
        # key乘上权重得到一个头（但是同时计算多头） B*L*H*d
        key = self.w_k_s(key).view(batch_size, seq_len_k, param.heads, param.d_k)
        # value乘上权重得到一个头（但是同时计算多头） B*L*H*d
        value = self.w_v_s(value).view(batch_size, seq_len_v, param.heads, param.d_v)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_q, param.d_q)  # query的维度融合变换 (B*H)*L*d
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_k, param.d_k)  # key的维度融合变换 (B*H)*L*d
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_len_v, param.d_v)  # value的维度融合变换 (B*H)*L*d

        # 产生多头的mask
        pad_mask = mask.repeat(param.heads, 1, 1) # (B*h)*L*L

        attn = self.attention(query, key, value, pad_mask)  # 把全部的头送入scaled dot-product attn中计算

        attn = attn.view(param.heads, batch_size, seq_len_q, param.d_v)  # attn的融合变换 h*B*L*d
        attn = attn.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len_q, -1)  # B*T*(h*d=d_model)

        attn = self.dropout(self.concat_heads(attn))  # 级联多头 B*L*(h*d=d_model)

        attn = self.layer_norm(residual + attn)  # 残差+LayerNorm

        return attn


class PositionalEncoding(nn.Module):
    """
    由于没有使用传统的基于RNN或者CNN的结构，那么输入的序列就无法判断其顺序信息，但这序列来说是十分重要的，
    Positional Encoding的作用就是对序列中词语的位置进行编码，这样才能使模型学会顺序信息。
    使用正余弦函数编码位置，pos在偶数位置为正弦编码，奇数位置为余弦编码。
    PE(pos, 2i)=sin(pos/10000^(2i/d_model)
    PE(pos, 2i+1)=cos(pos/10000^2i/d_model)
    即给定词语位置，可以编码为d_model维的词向量，位置编码的每一个维度对应正弦曲线
    上面表现出的是位置编码的绝对位置编码（即只能区别前后位置，不能区别前前前或者后后后位置）
    又因为正余弦函数能表达相对位置信息，即
    sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
    cos(a+b)=cos(a)cos(b)-sin(a)sin(b)
    对于词汇之间的位置偏移k，PE(pos+k)可以表示成PE(pos)+PE(k)组合形式，
    那么就能表达相对位置（即能区分长距离的前后）
    """
    def __init__(self, max_seq_len, d_model, pad_idx):
        """
        位置编码
        :param max_seq_len: 序列的最大长度
        :param d_model: 模型的维度
        :param pad_idx: 填充符位置，默认为0
        """
        super(PositionalEncoding, self).__init__()
        pos_enc = np.array([
            [pos / np.power(10000, 2.0 * (i // 2) / d_model) for i in range(d_model)]
            for pos in range(max_seq_len)])  # 构建位置编码表

        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # sin计算偶数位置
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # cos计算奇数位置

        pos_enc[pad_idx] = 0.  # 第一行默认为pad符，全部填充0

        self.pos_embedding = nn.Embedding(max_seq_len, d_model)  # 设置位置向量层 L*D
        # 载入位置编码表，并不更新位置编码层
        self.pos_embedding.from_pretrained(torch.FloatTensor(pos_enc), freeze=True)

    def forward(self, src_seq):
        # 返回该批序列每个序列的字符的位置编码embedding  # T
        return self.pos_embedding(src_seq.to(device=param.device))


# Encoder的一层
class EncoderLayer(nn.Module):
    """
    encoder的一层由两个子层构成，multi-head attention+FFN
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multi_heads_attn = MultiHeadAttention()  # 编码器子层一，多头注意力机制层
        self.feed_forward = PositionWiseFeedForward()  # 编码器子层二，按位置的前馈层

    def forward(self, enc_in, pad_mask, no_pad_mask):
        """
        :param enc_in: 输入批次序列 B*L
        :param pad_mask: 屏蔽位B*L*L
        :param no_pad_mask: 屏蔽位B*L*1
        :return: B*L*(h*d=d_model)
        """
        attn = self.multi_heads_attn(enc_in, enc_in, enc_in, pad_mask)  # 子层一，多头注意力层计算 B*L*(h*d=d_model)
        attn *= no_pad_mask  # 进一步屏蔽掉补齐位 B*L*(h*d=d_model)

        output = self.feed_forward(attn)  # 子层二，按位置前馈层计算 B*L*(h*d=d_model)
        output *= no_pad_mask  # 进一步屏蔽掉补齐位 B*L*(h*d=d_model)

        return output


# Encoder的计算
class Encoder(nn.Module):
    """
    Encoder由6层EncodeLayer堆叠构成
    """
    def __init__(self, input_vocab_num, max_seq_len, pad_idx=0):
        """
        :param input_vocab_num: 全部输出序列的词典的单词数
        :param max_seq_len: 输入序列最大长度
        :param pad_idx: pad的填充位置，默认为0
        """
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(input_vocab_num, param.d_model, padding_idx=pad_idx)  # 词向量层 N*D
        self.pos_encoding = PositionalEncoding(max_seq_len + 1, param.d_model, pad_idx)  # 位置向量层 (N+1)*D
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(param.layers)])  # 堆叠n层encoder_layer
        self.pad_obj = Mask()  # mask对象

    def forward(self, src_seq):
        """

        :param src_seq: 输入批序列 B*L
        :return: 6层encoder子层的计算结果 B*L*？
        """
        src_pos = tool.seq_2_pos(src_seq)  # 生成输入序列对应的位置序列 B*L
        # Encoder第一层的输入是词向量word embedding + 位置向量positional encoding B*L*D
        enc_in = self.word_embedding(src_seq) + self.pos_encoding(src_pos)

        pad_mask = self.pad_obj.padding_mask(src_seq, src_seq)  # pad_mask 由补齐序列产生的屏蔽位 B*L*L
        no_pad_mask = self.pad_obj.no_padding_mask(src_seq)  # 序列补齐位的屏蔽位 B*L*1

        enc_out = 0
        for encoder_layer in self.encoder_layers:  # 循环计算每一层
            enc_out = encoder_layer(enc_in, pad_mask, no_pad_mask)
            enc_in = enc_out  # 上一层的输出等于下一层的输入

        return enc_out


# Decoder的一层
class DecoderLayer(nn.Module):
    """
    decoder的一层三个子层构成，mask multi-head attention + multi-head attention + FFN
    """
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.mask_multi_head_attn = MultiHeadAttention()  # 解码器子层一，带mask的多头注意力机制层
        self.multi_head_attn = MultiHeadAttention()  # 解码器子层二，多头注意力机制层
        self.feed_forward = PositionWiseFeedForward()  # 解码器子层三，按位置前馈层

    def forward(self, dec_in, enc_out, no_mask, pad_seq_mask, dec_enc_mask):
        """

        :param dec_in: 目标批序列 B*
        :param enc_out: 编码输出 B*T*?
        :param no_mask: 序列补齐位的屏蔽位 B*T*1
        :param pad_seq_mask: 序列补齐位的屏蔽位 B*T*T
        :param dec_enc_mask: 编码-解码屏蔽位 B*T*T
        :return: 一层解码计算的结果 B*T*(d*h=d_model)
        """
        attn = self.mask_multi_head_attn(dec_in, dec_in, dec_in, pad_seq_mask)  # 子层一，mask多头注意力层计算 B*L*(d*h)
        attn = attn * no_mask if no_mask is not None else attn  # 屏蔽序列补齐位 B*L*(d*h)

        attn = self.multi_head_attn(attn, enc_out, enc_out, dec_enc_mask)  # 子层二，多头（交互）注意力层计算 B*L*(d*h)
        attn = attn * no_mask if no_mask is not None else attn  # 屏蔽序列补齐位 B*L*(d*h)

        out = self.feed_forward(attn)  # 子层三，按位置前馈层计算 B*L*(d*h)
        out = out * no_mask if no_mask is not None else out  # 屏蔽序列补齐位 B*L*(d*h)

        return out


# Decoder的计算
class Decoder(nn.Module):
    """
    Decoder由6层DecoderLayer构成
    """
    def __init__(self, target_vocab_num, max_seq_len, pad_idx=0):
        """

        :param target_vocab_num:
        :param max_seq_len:
        :param pad_idx:
        """
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(target_vocab_num, param.d_model)  # 构建词向量层 M*D
        self.pos_encoding = PositionalEncoding(max_seq_len + 1, param.d_model, pad_idx)  # 构建位置向量层 (M+1)*D
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(param.layers)])  # 堆叠6层DecoderLayer
        self.mask_obj = Mask()

    def forward(self, tgt_seq, src_seq, enc_out):
        """

        :param tgt_seq: 该批目标序列 B*L
        :param src_seq: 该批输入序列 B*L
        :param enc_out: 编码器的输出 B*L*(d*h=d_model)
        :return: 解码器的输出 B*L*(d*h)
        """
        tgt_pos = tool.seq_2_pos(tgt_seq)  # 生成目标序列的位置向量 B*L
        no_pad_mask = self.mask_obj.no_padding_mask(tgt_seq)  # 生成target序列补齐位的屏蔽位 B*L*1

        pad_mask = self.mask_obj.padding_mask(tgt_seq, tgt_seq)  # 生成序列的补齐位的屏蔽位 B*L*L
        seq_mask = self.mask_obj.sequence_mask(tgt_seq)  # 生成子序列屏蔽位（上三角形） B*L*L
        pad_seq_mask = (pad_mask + seq_mask).gt(0)  # 在解码器中，结合两种mask B*L*L

        # 在第二层的多头注意力机制中，产生context类的mask B * tgt_L * src_L
        dec_enc_mask = self.mask_obj.padding_mask(src_seq, tgt_seq)

        # Decoder的第一层为词向量word embedding+位置向量 embedding
        dec_in = self.word_embedding(tgt_seq) + self.pos_encoding(tgt_pos)  # B*L*(h*d=d_model)

        dec_out = 0
        for decoder_layer in self.decoder_layers:  # 循环计算每一层
            dec_out = decoder_layer(dec_in, enc_out, no_pad_mask, pad_seq_mask, dec_enc_mask)
            dec_in = dec_out  # 上一层的输出等于下一层的输入

        return dec_out


# Transformer的实现类
class Transformer(nn.Module):
    """
    Transformer由Encoder和Decoder构成
    """
    def __init__(self, input_vocab_num, target_vocab_num, src_max_len, tgt_max_len):
        """
        Transformer模型的主类
        :param input_vocab_num: 全部输入序列的词典的单词数
        :param target_vocab_num: 全部目标序列的词典的单词数
        :param src_max_len: 全部输入序列的最大长度
        :param tgt_max_len: 全部目标序列的最大长度
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_vocab_num, src_max_len)
        self.decoder = Decoder(target_vocab_num, tgt_max_len)
        self.word_prob_map = nn.Linear(param.d_model, target_vocab_num, bias=False)  # 最后的线性转换层 D*M
        nn.init.xavier_normal_(self.word_prob_map.weight)  # 初始化线性层权重分头
        # self.softmax = nn.LogSoftmax(dim=0)  # 最后的softmax层 使用交叉熵，就不用做softmax了

    def forward(self, src_seq, tgt_seq):
        """

        :param src_seq: 输入批次序列 B*L
        :param tgt_seq: 目标批次序列 B*L
        :return: 映射到目标序列单词表的输出 B*L*M
        """

        tgt_seq = tgt_seq[:, :-1]  # TODO 这里为什么要截断最后一个

        enc_out = self.encoder(src_seq)  # 编码器编码输入序列 B*L*(h*d=d_model)

        dec_out = self.decoder(tgt_seq, src_seq, enc_out)  # 解码器解码输入序列（训练时也计算目标序列）

        dec_out = self.word_prob_map(dec_out)  # 映射到全部的单词 B*L*M

        output = dec_out  # 使用的是交叉熵，那就不需要做softmax了

        pre = output.view(-1, output.size(2))  # 维度变换，为了下一步的loss的计算 (B*L)*M

        return pre


# Mask操作的实现类
class Mask():

    def __init__(self):
        pass

    # 对序列做补齐的位置
    def padding_mask(self, seq_k, seq_q):
        """
        生成padding masking TODO 待解释seq_k和seq_q的关系和来源
        :param seq_k: B*L
        :param seq_q: B*L
        :return: 产生B*T*T的pad_mask输出
        """
        seq_len = seq_q.size(1)
        pad_mask = seq_k.eq(param.pad)  # 通过比较产生pad_mask B*T

        return pad_mask.unsqueeze(1).expand(-1, seq_len, -1)

    # 不对序列做补齐的位置
    def no_padding_mask(self, seq):
        """
        pad_mask的反向操作
        :param seq: B*T
        :return: B*T*T
        """
        return seq.ne(param.pad).type(torch.float).unsqueeze(-1)

    # 序列屏蔽，用于decoder的操作中
    def sequence_mask(self, seq):
        """
        屏蔽子序列信息，防止decoder能解读到，使用一个上三角形来进行屏蔽
        seq: B*T batch_size*seq_len
        :return: seq_mask B*T*T batch_size*seq_len*seq_len
        """
        batch_size, seq_len = seq.shape
        # 上三角矩阵来屏蔽不能看到的子序列
        seq_mask = torch.triu(
             torch.ones((seq_len, seq_len), device=param.device, dtype=torch.uint8), diagonal=1)

        return seq_mask.unsqueeze(0).expand(batch_size, -1, -1)


# 损失函数计算类
class Criterion():

    def cal_loss(self, real_tgt, pre_tgt):
        loss = F.cross_entropy(pre_tgt, real_tgt, ignore_index=param.pad, reduction=param.loss_cal)  # 计算损失

        pre = pre_tgt.max(1)[1]
        real = real_tgt.contiguous().view(-1)
        non_pad_mask = real.ne(param.pad)
        correct = pre.eq(real)
        correct = correct.masked_select(non_pad_mask).sum().item()

        return loss, correct


# 特殊学习率下的优化器
class SpecialOptimizer():

    def __init__(self, optimizer, warmup_steps, d_model, step_num=0):
        """
        随着训练步骤，学习率改变的模型优化器
        :param optimizer: 预定义的优化器
        :param warmup_steps: 预热步
        :param d_model: 模型维度
        :param step_num: 当前的模型的的训练步数
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = step_num

    def zero_grad(self):  # 优化器梯度清零
        self.optimizer.zero_grad()

    def step_update_lrate(self):  # 优化器更新学习率和步进
        self.step_num += 1  # 批次训练一次，即为步数一次，自加1

        # 生成当前步的学习率
        lr = np.power(self.d_model, -0.5) * np.min([np.power(self.step_num, -0.5),
                                                    np.power(self.warmup_steps, -1.5) * self.step_num])

        # 把当前步的学习率赋值给优化器
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()  # 优化器步进

        return lr


# 引用他人的推理过程
class Translator(object):

    # TODO 后期要设置存储已经训练好的模型，推理预测时再载入
    def __init__(self, model, tgt_max_len):
        self.model = model.to(param.device)

        self.word_prob_prj = nn.LogSoftmax(dim=1)  # 解码生成词的映射层

        self.model.eval()  # 设置模型为验证模型，不反向传播，更新参数

        self.tgt_max_len = tgt_max_len

    def translate_batch(self, src_seq, src_pos):
        ''' Translation work in one batch 按批次解码生成序列'''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. 指示示例在张量中的位置'''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            """
            Collect tensor parts associated to active instances. 收集与活动实例关联的张量部分
            :param beamed_tensor: 输入序列 [batch_size * beam_size, input_max_len]
            :param curr_active_inst_idx: 目前激活的实例索引 一维大小为batch_size的tensor
            :param n_prev_active_inst: n个之前激活的示例 batch_size
            :param n_bm: beam_size大小
            :return:
            """

            _, *d_hs = beamed_tensor.size()  # 批输入序列的最大序列长度
            n_curr_active_inst = len(curr_active_inst_idx)  # 当前的激活实例数 batch_size
            new_shape = (n_curr_active_inst * n_bm, *d_hs)  # 构造 batch_size * input_max_len的最大元祖

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)  # beamed_tensor变化的形状 batch_size * -1
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)  # 选择激活的索引
            beamed_tensor = beamed_tensor.view(*new_shape)  # 转变形状为 [batch_size * beam_size, input_max_len]

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            """
            Sentences which are still active are collected, 收集已经完成的句子
            so the decoder will not run on completed sentences. 不会在不完成的句子上继续执行
            :param src_seq: 输入序列 [batch_size * beam_size, input_max_len]
            :param src_enc: 编码器输出 [batch_size * beam_size, input_max_len, dimension]
            :param inst_idx_to_position_map:索引到位置的匹配字典 {0:0, 1:1, ..., batch_size - 1:batch_size - 1}
            :param active_inst_idx_list: 激活的索引列表 [0, 1, 2, ..., batch_size-1]
            :return:
            """
            n_prev_active_inst = len(inst_idx_to_position_map)  # 获取字典大小 batch_size
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(param.device)  # 转化为LongTensor类型

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            """
            :param inst_dec_beams: 解码的beam search对象 batch size 个
            :param len_dec_seq: 解码的序列位置，从1~最大长度+1？
            :param src_seq:输入序列  B*L
            :param enc_output:编码器的编码输出 B*L*D
            :param inst_idx_to_position_map: 索引到位置的匹配字典，{0:0，1:1,...,batch_size:batch_size}
            :param n_bm: beam search的beam size大小
            :return:
            """
            ''' Decode and update beam status, and then return active beam idx 解码并更新beam状态，然后返回激活的beam索引 '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                # 解码的偏置序列？是一个列表，共有batch_size个元素，每个元素是[beam，1]的矩阵，第一个值为开始符2
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                # 把解码的部分序列堆叠起来，变为[batch_size, beam_size, 1]的矩阵
                dec_partial_seq = torch.stack(dec_partial_seq).to(param.device)
                # 变换为[batch_size*beam_size, 1]的矩阵
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                """
                准备解码序列的pos位置向量
                :param len_dec_seq: 解码序列的长度？
                :param n_active_inst: 需要激活的句子，大小为batch_size
                :param n_bm: beam_size
                :return:
                """
                # 解码部分位置向量 [1] 一个大小为[1]，值为1的tensor
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=param.device)
                # 解码部分位置向量 扩展为其大小为[batch_size*beam_size, 1]， 每个值都是1
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
                """
                预测出来额单词
                :param dec_seq: 解码序列 [beam_size*batch_size, 1] beam_size为一个循环，第一个值为开始符2
                :param dec_pos: 解码序列为位置向量 [batch_size*batch_size, 1] 值都为1
                :param src_seq: 输入序列 [batch_size*beam_size, input_max_len]
                :param enc_output: 编码器输出 [batch_size*beam_size, input_max_lem, dimension]
                :param n_active_inst: 需要记录的序列数量 == batch_size
                :param n_bm: beam_size
                :return:
                """
                # 解码器解码输出 [batch_size*beam_size, 1，d]
                dec_output = self.model.decoder(dec_seq, src_seq, enc_output)  # 解码器输出
                # 选择最后一步的解码序列, 本来也只有一步 [batch_size*beam_size, d]
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                # 所有词进行映射输出，得到每个词的概率 [batch_size*beam_size, M]
                word_prob = F.log_softmax(self.model.word_prob_map(dec_output), dim=1)
                # 变换形状 [batch_size, beam_size, M]
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                """

                :param inst_beams: beam search 对象 batch_size 个
                :param word_prob: 模型预测出来的结果 [batch_size, beam_size, M]
                :param inst_idx_to_position_map: 索引到位置的匹配字典 {0:0,1:1,2:2,...,batch_size-1:batch_size-1}
                :return:
                """
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():  # 循环索引和位置
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:  # 如果没有完成
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)  # n个激活的batch 值为batch_size

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)  # 准备beam search的目标序列
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)  # 准备beam search的目标序列POS位置
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            # 用预测的单词概率信息更新beam search并收集不完整的实例？？？
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #-- Encode  先编码
            src_seq, src_pos = src_seq.to(param.device), src_pos.to(param.device)
            src_enc = self.model.encoder(src_seq)

            #-- Repeat data for beam search
            n_bm = param.beam_size  # beam search个数多少
            n_inst, len_s, d_h = src_enc.size()  # 批大小，该批序列最大长度，维度
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)  # 重复生成n次src_seq [n*b, l]
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)  # 重复生成n次src_seq [n*b, h, d]

            #-- Prepare beams 准备beam search的对象，对象个数和batch size一致
            inst_dec_beams = [Beam(n_bm, device=param.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not  是否有效记账？？？
            active_inst_idx_list = list(range(n_inst))  # 批大小的list列表[0,2,...,batch_size]
            # 返回一个字典，{0:0,1:1,...,batch_size:batch_size}
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode 解码阶段
            for len_dec_seq in range(1, self.tgt_max_len + 1):  # 在1~序列最大长度+1之内循环？为什么

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, param.beam_search_left)

        return batch_hyp, batch_scores


# 引用他人的beam search
class Beam():
    ''' Beam search beam搜索'''

    def __init__(self, size, device=False):

        self.size = size
        self._done = False

        # The score for each translation on the beam.收集每一层的评分
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)  # 大小为beam size的一维tensor
        self.all_scores = []

        # The backpointers at each time-step.  # TODO 这个变量是啥意思
        self.prev_ks = []

        # The outputs at each time-step. 每一步的输出
        # 列表，里面的元素是长度为beam size，值为0的一维tensor
        self.next_ys = [torch.full((size,), param.pad, dtype=torch.long, device=device)]
        self.next_ys[0][0] = param.sos  # list第一个元素为一维大小beam size的tensor，第一个值为开始符BOS

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        """
        Update beam status and check if finished or not. 更新beam状态并检查是否完成
        :param word_prob: [beam_size, M] 预测出来所有词的概率
        :return:
        """
        num_words = word_prob.size(1)  # 目标序列所有单词数

        # Sum the previous scores.  把之前所有的分数相加
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)  # 变换为一维矩阵，大小为 M

        # 获取其中top beam_size项的概率和索引位置
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array, 变换 BestScoreSid beam*word词数的数组，
        # so we need to calculate which word and beam each score came from 计算每个单词和beam 搜索分数的来源
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.  结束条件是每个beam的顶端为EOS结束符
        if self.next_ys[-1][0].item() == param.eos:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep. 获取当前时间步的解码序列"

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)  # 初始解码序列 [beam_size, 1]，第一个为开始符
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[param.sos] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


# 总的计算模型，序列到序列
class Sequence2Sequence():
    """
    模型整体的实现类
    """
    def __init__(self, transformer, optimizer, criterion):
        """

        :param src_vocab_num:b 输入全部序列的词典的单词数
        :param tgt_vocab_num: 目标全部序列词典的单词数
        :param src_max_len: 输入全部序列的词典的单词数
        :param tgt_max_len: 目标全部序列的词典的单词数
        :param criterion: 损失函数
        :param optimizer: 模型优化器
        """
        self.transformer = transformer  # transformer模型
        self.optimizer = optimizer  # 优化器
        self.criterion = criterion  # 损失函数

    # 训练-验证
    def train_val(self, train_loader, val_loader):
        print('#=======开始训练&验证==========#\n')
        total_step = 0  # 计算总的步数
        step = 0
        for epoch in range(param.epochs):  # 按轮次训练数据
            print('[轮次%3.0f]' % epoch)

            each_loss = []
            self.transformer.train()  # 设置模型为训练状态
            total_loss = 0  # 本轮次所有批次数据的训练损失
            total_word_num = 0  # 本轮次所有批次数据的词数
            total_word_correct = 1  # 本轮次所有批次数据中单词正取的个数

            epoch_start_time = datetime.datetime.now()  # 一个轮次模型的计算开始时间
            for step, batch_data in tqdm(enumerate(train_loader), desc=' 训练', leave=False):  # 迭代计算批次数据
                batch_start_time = datetime.datetime.now()  # 一个批次计算的开始时间
                total_step += 1

                self.optimizer.zero_grad()  # 优化器梯度清零
                src_seq, tgt_seq = tool.batch_2_tensor(batch_data)  # 得到输入和目标序列 B*L B*L
                self.transformer.train()
                pre_tgt = self.transformer(src_seq, tgt_seq)
                real_tgt = tgt_seq[:, 1:].contiguous().view(-1) # 实际的目标序列token
                # 采用交叉熵损失函数，那么最后一步就不需要做softmax了！！！
                loss, correct = self.criterion.cal_loss(real_tgt, pre_tgt)
                loss.backward(retain_graph=True)  # 损失反向传播(计算损失后还保留变量)
                learn_rate = self.optimizer.step_update_lrate()  # 更新学习率，优化器步进

                total_loss += loss.item()
                each_loss.append(loss.detach())

                non_pad_mask = real_tgt.ne(param.pad)
                word_num = non_pad_mask.sum().item()
                total_word_num += word_num
                total_word_correct += correct

                batch_end_time = datetime.datetime.now()
                if param.show_loss:
                    print('|该批数 %4.0f' % step,  # 当前轮次第几个批次
                          '|总批数 Step %8.0f' % total_step,  # 总的批次数
                          '|批大小 %3.0f' % len(batch_data[0]),  # 当前批次大小
                          '|损失 %7.5f' % loss.detach(),  # 当前批次损失
                          '|该批耗时', batch_end_time - batch_start_time,
                          '|批学习率 %15.14f' % learn_rate)  # 当前批次学习率大小

            loss_per_word = total_loss / total_word_num  # 平均到每个单词的损失
            ppl = math.exp(min(loss_per_word, 100))  # 困惑度，越小越好
            acc = total_word_correct / total_word_num  # 平均到每个单词的准确率acc
            acc = 100 * acc  # 准确率，越大越好
            epoch_end_time = datetime.datetime.now()

            print('批数 %4.0f' % (step+1),
                  '| 累积批数%8.0f' % total_step,
                  '| 批大小%3.0f' % param.batch_size,
                  '| 耗时', epoch_end_time - epoch_start_time)

            print('训练',
                  '| 困惑度↓PPL %8.6f' % ppl,
                  '| 准确率↑ACC %8.5f' % acc,
                  '| 首批损失 %7.5f' % each_loss[0],
                  '| 尾批损失 %7.5f' % each_loss[-2])

            self.evaluate(val_loader, epoch)  # 每一个训练轮次结束，验证一次

        # TODO 训练结束，待保存训练模型
        print('训练&验证结束！')

    # 每个轮次训练结束，用验证数据验证一次模型
    def evaluate(self, data_loader, epoch):
        self.transformer.eval()  # 设置模型为验证状态
        total_loss = 0  # 本轮次所有批次数据的训练损失
        total_word_num = 0  # 本轮次所有批次数据的词数
        total_word_correct = 0  # 本轮次所有批次数据中单词正取的个数

        with torch.no_grad():  # 设置训练产生的损失不更新模型
            for step, batch_data in tqdm(enumerate(data_loader), desc=' 验证--', leave=False):  # 迭代计算批次数据
                src_seq, tgt_seq = tool.batch_2_tensor(batch_data)  # 获取输入序列和验证序列
                pre_tgt = self.transformer(src_seq, tgt_seq)  # 模型预测的目标序列
                real_tgt = tgt_seq[:, 1:].contiguous().view(-1)  # 构建真实的目标序列
                loss, correct = self.criterion.cal_loss(real_tgt, pre_tgt)

                total_loss += loss.item()

                non_pad_mask = real_tgt.ne(param.pad)
                word_num = non_pad_mask.sum().item()
                total_word_num += word_num
                total_word_correct += correct

                if param.show_loss:
                    print('验证批次数 % 3.0f' % step,
                          '| 批大小 %3.0f' % len(batch_data[0]),
                          '| 损失 %7.5f' % loss.detach())

            loss_per_word = total_loss / total_word_num  # 平均到每个单词的损失
            ppl = math.exp(min(loss_per_word, 100))  # 困惑度，越小越好
            acc = total_word_correct / total_word_num  # 平均到每个单词的准确率acc
            acc = 100 * acc  # 准确率，越大越好

            print('验证',
                  '| 困惑度↓PPL %8.6f' % ppl,
                  '| 准确率↑ACC %8.5f' % acc)
                  # '-首批损失-%7.5f' % each_loss[0],
                  # '-尾批损失-%7.5f' % each_loss[-2])

    # 批序列推理过程
    def inference(self, data_loader, source_lang, target_lang, tgt_max_len):
        print('#=======开始推理测试==========#\n')

        def index_2_word(lang, seq):
            """ 转化索引到单词"""
            seq = [int(idx.detach()) for idx in seq]
            new_seq = []
            for i in seq:
                if i != param.sos and i != param.eos and i != param.pad:
                    new_seq.append(i)
            idx_2_word = [lang.index2word[i] for i in new_seq]

            return idx_2_word

        infer = Translator(self.transformer, tgt_max_len)  # 批次数据推理类

        with open('./input_target_infer.txt', 'w', encoding='utf-8') as f:

            for step, batch_dat in tqdm(enumerate(data_loader), desc='推理测试开始', leave=True):  # 迭代推理批次数据
                src_seq, tgt_seq = tool.batch_2_tensor(batch_dat)  # 获得输入序列和实际目标序列
                src_pos = tool.seq_2_pos(src_seq)  # 得到输入序列的pos位置向量
                all_pre_seq, all_pre_seq_p = infer.translate_batch(src_seq, src_pos)  # 获得所有预测的结果和对应的概率

                for index, pre_seq in enumerate(all_pre_seq):
                    src_word_seq = index_2_word(source_lang, src_seq[index])
                    tgt_word_seq = index_2_word(target_lang, tgt_seq[index])
                    for seq in pre_seq:
                        new_seq = []
                        for i in seq:
                            if i != param.sos and i != param.eos and i != param.pad:
                                new_seq.append(i)
                        pre_word_seq = [target_lang.index2word[idx] for idx in new_seq]

                    f.write('输入序列->：' + ' '.join(src_word_seq) + '\n')  # 写入输入序列
                    f.write('->预测序列：' + ' '.join(pre_word_seq) + '\n')  # 写入预测序列
                    f.write('==目标序列：' + ' '.join(tgt_word_seq) + '\n\n')  # 写入实际序列

        print('推理预测序列完毕！')


def main(train_src, train_tgt, val_src, val_tgt, test_src, test_tgt):
    #========================准备数据============================#
    data_obj = DataProcess(train_src, train_tgt, val_src, val_tgt, test_src, test_tgt)  # 训练数据对象
    src_tgt, src_lang, tgt_lang = data_obj.get_src_tgt_data()
    *_, src_tgt_seq_train = data_obj.word_2_index('train', src_lang, tgt_lang)  # 训练数据
    *_, src_tgt_seq_val = data_obj.word_2_index('val', src_lang, tgt_lang)  # 验证数据
    *_, src_tgt_seq_test = data_obj.word_2_index('test', src_lang, tgt_lang)  # 测试数据

    # 打包批次数据
    train_data_loader = DataLoader(src_tgt_seq_train, param.batch_size, True, drop_last=False)
    val_data_loader = DataLoader(src_tgt_seq_val, param.batch_size, False, drop_last=False)
    test_data_loader = DataLoader(src_tgt_seq_test, param.infer_batch, True, drop_last=False)  # 批序列推理预测

    # ========================定义模型============================#
    transformer = Transformer(  # 定义transformer模型
        input_vocab_num=src_lang.n_words,
        target_vocab_num=tgt_lang.n_words,
        src_max_len=data_obj.src_max_len,
        tgt_max_len=data_obj.tgt_max_len).to(param.device)

    optimizer = SpecialOptimizer(  # 定义优化器， 返回优化器类的对象
        optimizer=torch.optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()), betas=(0.9, 0.98), eps=1e-09),
        warmup_steps=param.warmup_step,
        d_model=param.d_model)

    criterion = Criterion()  # 定义损失函数

    # 定义总的计算模型，开始训练（验证）和推理
    seq2seq = Sequence2Sequence(transformer= transformer, optimizer=optimizer, criterion=criterion)

    #========================训练（验证）模型=====================#
    seq2seq.train_val(train_data_loader, val_data_loader)

    #========================模型推理测试===========================#
    seq2seq.inference(test_data_loader, src_lang, tgt_lang, data_obj.tgt_max_len)


if __name__ == '__main__':
    # 训练数据集
    train_source = os.path.abspath('..') + '/data/train.en'
    train_target = os.path.abspath('..') + '/data/train.de'

    # 验证数据集
    val_source = os.path.abspath('..') + '/data/val.en'
    val_target = os.path.abspath('..') + '/data/val.de'

    # 测试数据集
    test_source = os.path.abspath('..') + '/data/test.en'
    test_target = os.path.abspath('..') + '/data/test.de'

    main(train_source, train_target, val_source, val_target, test_source, test_target)
