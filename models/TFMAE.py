'''
Ya, S., Youjian, Z., Chenhao, N., Rong, L., Wei, S., & Dan, P. (2019) 
Robust Anomaly Detection for Multivariate Time Series Through Stochastic 
Recurrent Neural Network, ACM SIGKDD Conference on Knowledge Discovery 
and Data Mining: 2828-2837.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入注意力层和嵌入层
from src.MTFAEutils import *

class Encoder(nn.Module):
    """
    编码器模块：由多个注意力层组成，用于特征提取
    """

    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        # 注意力层列表
        self.attn_layers = nn.ModuleList(attn_layers)
        # 归一化层（可选）
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x: 输入张量，形状为[B, T, D]，其中B=批次大小，T=时间步长，D=特征维度
        attlist = []  # 存储各层注意力权重
        for attn_layer in self.attn_layers:
            # 通过注意力层，获取输出和注意力权重
            x, _ = attn_layer(x)
            attlist.append(_)  # 保存注意力权重

        # 应用最终归一化（如果有）
        if self.norm is not None:
            x = self.norm(x)

        return x, attlist  # 返回编码后的特征和注意力权重列表


class FreEnc(nn.Module):
    """
    频率域编码器：在频率域进行掩码自编码
    """

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, fr):
        super(FreEnc, self).__init__()
        # 数据嵌入层：将输入特征映射到模型维度并添加位置信息
        self.emb = DataEmbedding(c_in, d_model)

        # 解码器：由多个注意力层组成
        self.enc = Encoder(
            [
                AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # 投影层：将编码器输出转换为注意力权重
        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # 使用sigmoid确保输出在0-1范围内
        )

        # 频率域掩码标记：用于替换被掩码的频率分量（复数类型）
        self.mask_token = nn.Parameter(torch.zeros(1, d_model, 1, dtype=torch.cfloat))

        # 频率域掩码比例
        self.fr = fr

    def forward(self, x):
        # x: 输入时间序列，形状为[B, T, C]，其中C=输入特征数

        # 特征嵌入：将输入映射到模型维度并添加位置信息
        ex = self.emb(x)  # 输出形状: [B, T, D]

        # 转换到频率域并计算幅度
        # 先转置为[B, D, T]，再进行FFT（快速傅里叶变换）
        # cx = torch.Size([64, 128, 51])
        cx = torch.fft.rfft(ex.transpose(1, 2))
        # 计算复数的幅度 (实部平方+虚部平方的平方根)
        # mag = torch.Size([64, 128, 51])
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2)  # 输出形状: [B, D, Mag]，Mag为频率分量数量

        # 掩码较小的幅度分量
        # 计算每个特征维度上的幅度分位数，用于确定掩码阈值
        quantile = torch.quantile(mag, self.fr, dim=2, keepdim=True)
        # 找到幅度小于分位数的位置索引
        idx = torch.argwhere(mag < quantile)
        # 用掩码标记替换这些位置的频率分量
        cx[mag < quantile] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[idx[:, 0], idx[:, 1], idx[:, 2]]

        # 转换回时间域（逆傅里叶变换）
        ix = torch.fft.irfft(cx).transpose(1, 2)  # 输出形状: [B, T, D]

        # 编码处理后的时间序列
        dx, att = self.enc(ix)

        # 投影得到最终注意力权重
        rec = self.pro(dx)
        att.append(rec)

        return att  # 返回注意力权重列表 [B, T, T]


class TemEnc(nn.Module):
    """
    时间域编码器：在时间域进行掩码自编码
    """

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, tr):
        super(TemEnc, self).__init__()
        # 数据嵌入层
        self.emb = DataEmbedding(c_in, d_model)
        # 位置嵌入层
        self.pos_emb = PositionalEmbedding(d_model)

        # 编码器：处理未掩码的时间点
        self.enc = Encoder(
            [
                AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 解码器：重建完整序列
        self.dec = Encoder(
            [
                AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 投影层：生成注意力权重
        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # 时间域掩码标记
        # nn.Parameter将mask_token添加为可学习的参数
        # 生成(1,1,128)的掩码
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # 计算需要掩码的时间点数量（总长度 * 掩码比例）
        self.tr = int(tr * win_size)
        # 序列窗口大小，用于计算统计特征
        self.seq_size = seq_size

    def forward(self, x):
        # x: 输入时间序列，形状为[B, T, C]

        # 特征嵌入
        ex = self.emb(x)  # ex: [B, T, D]

        # 计算输入特征的平方，用于后续方差计算
        ex2 = ex ** 2     # ex2: [B, T, D]

        # 获取设备信息
        device = ex.device

        # 定义卷积核，用于计算滑动窗口内的统计特征（均值和平方和）
        # 对MSL： seq_size = 10
        filters = torch.ones(1, 1, self.seq_size).to(device)

        # 计算滑动窗口内的均值（通过卷积实现）
        # 先转置为[B*D, 1, T]，便于卷积计算
        ltr = F.conv1d(
            # ex [B,T,D] -> [B,D,T] -> [B*T,D] -> [B*D, 1, T]
            ex.transpose(1, 2).reshape(-1, ex.shape[1]).unsqueeze(1),
            filters,
            padding=self.seq_size - 1  # 保持输出长度与输入一致
        )
        # ltr(8192,1,109)
        # 对窗口内的求和结果进行归一化（前几个窗口长度不足，需要特殊处理）
        ltr[:, :, :self.seq_size - 1] /= torch.arange(1, self.seq_size).to(device)
        ltr[:, :, self.seq_size - 1:] /= self.seq_size

        # 计算滑动窗口内的平方和（用于计算方差）
        ltr2 = F.conv1d(
            ex2.transpose(1, 2).reshape(-1, ex.shape[1]).unsqueeze(1),
            filters,
            padding=self.seq_size - 1
        )
        # 同样进行归一化
        ltr2[:, :, :self.seq_size - 1] /= torch.arange(1, self.seq_size).to(device)
        ltr2[:, :, self.seq_size - 1:] /= self.seq_size

        # 计算窗口内的方差和均值
        # 方差 = E[x²] - (E[x])²
        ltrd = (ltr2 - ltr ** 2)[:, :, :ltr.shape[-1] - self.seq_size + 1].squeeze(1)
        ltrd = ltrd.reshape(ex.shape[0], ex.shape[-1], -1).transpose(1, 2)

        # 均值
        ltrm = ltr[:, :, :ltr.shape[-1] - self.seq_size + 1].squeeze(1)
        ltrm = ltrm.reshape(ex.shape[0], ex.shape[-1], -1).transpose(1, 2)

        # 计算分数：方差之和 / 均值之和（用于确定掩码位置，分数高的被认为不重要）
        score = ltrd.sum(-1) / ltrm.sum(-1)

        # 确定掩码和未掩码的时间点索引
        # 选择分数最高的tr个时间点进行掩码（这些被认为是不重要的）
        masked_idx, unmasked_idx = score.topk(self.tr, dim=1, sorted=False)[1], \
            (-1 * score).topk(x.shape[1] - self.tr, dim=1, sorted=False)[1]

        # 提取未掩码的标记
        unmasked_tokens = ex[torch.arange(ex.shape[0])[:, None], unmasked_idx, :]

        # 编码未掩码的标记
        ux, _ = self.enc(unmasked_tokens)

        # 生成掩码标记（添加位置信息）

        masked_tokens = self.mask_token.repeat(ex.shape[0], masked_idx.shape[1], 1) + self.pos_emb(idx=masked_idx)

        # 组合未掩码和掩码标记，重建序列
        tokens = torch.zeros(ex.shape, device=device)
        tokens[torch.arange(ex.shape[0])[:, None], unmasked_idx, :] = ux
        tokens[torch.arange(ex.shape[0])[:, None], masked_idx, :] = masked_tokens

        # 解码重建的序列
        dx, att = self.dec(tokens)


        rec = self.pro(dx)
        att.append(rec)

        return att  # 返回注意力权重列表 [B, T, T]


class MTFA(nn.Module):
    """
    主模型：结合时间域和频率域编码器的掩码自编码器
    """

    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(MTFA, self).__init__()
        # 全局设备变量
        global device
        device = dev

        # 初始化时间域编码器
        self.tem = TemEnc(
            c_in=c_in,
            c_out=c_out,
            d_model=d_model,
            e_layers=e_layers,
            win_size=win_size,
            seq_size=seq_size,
            tr=tr
        )

        # 初始化频率域编码器
        self.fre = FreEnc(
            c_in=c_in,
            c_out=c_out,
            d_model=d_model,
            e_layers=e_layers,
            win_size=win_size,
            fr=fr
        )

    def forward(self, x):
        # x: 输入时间序列，形状为[B, T, C]

        # 获取时间域注意力权重
        tematt = self.tem(x)  # tematt: [B, T, T]

        # 获取频率域注意力权重
        freatt = self.fre(x)  # freatt: [B, T, T]

        return tematt, freatt  # 返回两个域的注意力权重