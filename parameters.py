#-*-coding:utf-8-*-

# transformer的内置参数类


import torch


class Parameters():

    def __init__(self, d_mode=512, d_q=64, heads=8, d_ff=2048, dropout=0.1, layers=6):
        self.d_model = d_mode
        self.d_q = d_q
        self.d_k = d_q
        self.d_v = d_q
        self.heads = heads
        self.d_ff = d_ff
        # TODO d_model, heads, d_v, dff 四者的关系需要进行约束，简单来说就是d_model=heads*d_v，d_ff=2*d_model

        self.layers = layers

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = dropout

        self.pad = 0

        self.min_len = 0
        self.max_len = 100

        self.min_word_count = 5

        self.pad = 0
        self.unk = 1
        self.sos = 2
        self.eos = 3

        # 服务器上最大只能跑64*6=384的batch_size，还可以优化
        # 128-4000, 64-8000，32-16000，256-2000
        # 32-16,000; 64-8,000; 128-4,000; 256-2,000; 512-1,000; 1024-500
        self.batch_size = 128

        self.beam_search_left = 1

        self.epochs = 50

        self.use_gpu = True if torch.cuda.is_available() else False

        self.warmup_step = 4000

        self.loss_cal = 'sum'  #  损失是采用总和还是平均

        self.show_loss = False

        self.beam_size = 5

        self.infer_batch = 32

        # if !self.__check():
            # print('当前参数设置不满足条件，可能导致计算出错，请检查修改后再运行')
            # exit()

    def __check(self):
        """
        检查d_model，d_q，d_k，d_v，heads，d_ff，这几者之间关系
        :return: True/False
        """
        return False