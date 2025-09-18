'''
Shreshth, T., Giuliano, C., & Nicholas R., J. (2022) TranAD:
Deep Transformer Networks for Anomaly Detection in Multivariate 
Time Series Data., Proceedings of the VLDB Endowment, 15.6: 1201-1214.
'''

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from src.Transutils import PositionalEncoding
from src.constants import lr

class TranAD(nn.Module):
    def __init__(self, feats):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        # 编码层
        # d_model被设置为特征数量（feats）的两倍。使用一个二维的嵌入空间来表示每个输入特征。
        # dim_feedforward=16  前馈神经网络隐藏层维度，16个神经元
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        # 编码器，只使用上面定义的一层编码层
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        # 解码层1
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        # 解码器，只使用上面定义的一层解码层1
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        # 解码层2
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        # 解码器，只使用上面定义的一层解码层2
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        # 线性层
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # 拼接后 src:(10,128,76)
        src = src * math.sqrt(self.n_feats)  # 缩放src src:(10,128,76)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)  # memory:(10,128,76)
        tgt = tgt.repeat(1, 1, 2)  # tgt:(1,128,76)
        return tgt, memory

    # 调用：z = model(window, elem)
    def forward(self, src, tgt):  # tgt:(1,128,38)
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)  # c: (10,128,38) 初始条件c为零（无先验）
        # 解码器1输出O₁
        # *解包出tgt, memory
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))  # x1:(1,128,38)
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2  # c: (10,128,38)c更新为第一阶段重构误差
        # 解码器2输出Ô₂
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))  # x2:(1,128,38)
        return x1, x2