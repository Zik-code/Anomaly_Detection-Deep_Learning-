'''
Ling-rui, Y., Qiu-hong, L., & Yang, X. (2024) DTAAD: 
Dual Tcn-Attention Networks for Anomaly Detection in
 Multivariate Time Series Data, Knowledge-based systems, 295
'''

import math
import torch.nn as nn
from torch.nn import TransformerEncoder
from src.Transutils import PositionalEncoding,TransformerEncoderLayer
from src.constants import lr
from src.Transutils import Tcn_Local, Tcn_Global

class DTAAD(nn.Module):
    def __init__(self, feats):
        super(DTAAD, self).__init__()
        self.name = 'DTAAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)  # K=3&4 (Batch, output_channel, seq_len)
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers1 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)  # (seq_len, Batch, output_channel)
        encoder_layers2 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, num_layers=1)  # only one layer
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def callback(self, src, c):
        src2 = src + c # 原始数据src（128,38,10）与重构数据x1（128,38,1）广播相加，结果为 (128, 38, 10)
        g_atts = self.g_tcn(src2) # g_atts: (128, 38, 10)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        # src2: (10, 128, 38)（适配 Transformer)
        src2 = self.pos_encoder(src2)
        # memory 形状为 (10, 128, 38)
        memory = self.transformer_encoder2(src2)
        return memory

    def forward(self, src):
        l_atts = self.l_tcn(src)   # 输入src: (128,38,10), l_atts：(128,38,10)
        # (128,38,10)-> (10, 128, 38) （Transformer 要求输入格式为 (seq_len, batch_size, feature_dim)）
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        # 位置编码后数据结构不变
        src1 = self.pos_encoder(src1)
        # z1: (10, 128, 38)
        z1 = self.transformer_encoder1(src1)
        # 残差连接 self.fcn是线性层（nn.Linear(38, 38)），
        #  c1输出形状与z1相同 (10, 128, 38)
        c1 = z1 + self.fcn(z1)
        # (10, 128, 38) -> (128, 38, 10)
        # x1 ：(128, 38, 1) 第一阶段重构数据
        x1 = self.decoder1(c1.permute(1, 2, 0))
        z2 = self.fcn(self.callback(src, x1))
        # 残差连接
        c2 = z2 + self.fcn(z2)
        # x2 = (128, 38, 10)
        x2 = self.decoder2(c2.permute(1, 2, 0))
        #  返回(128, 1, 38)
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)  # (Batch, 1, output_channel)