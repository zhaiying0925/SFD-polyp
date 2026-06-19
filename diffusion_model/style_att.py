import torch
import torch.nn as nn
import torch.nn.functional as F

# class StyleExtractor(nn.Module):
#     """
#     使用通道注意力 + 全局均值方差提取风格特征
#     输入: style_image -> [B, C, H, W]
#     输出: style_feat -> [B, C]
#     """
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         # 通道注意力 (SE Block)
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(in_channels, in_channels // reduction)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(in_channels // reduction, in_channels)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         # 1. 全局平均池化
#         avg_feat = self.global_avg_pool(x).view(B, C)  # [B, C]
#
#         # 2. 通道注意力
#         scale = self.fc1(avg_feat)
#         scale = self.relu(scale)
#         scale = self.fc2(scale)
#         scale = self.sigmoid(scale)  # [B, C]
#
#         # 3. 应用到全局统计 (均值/方差)
#         mean = x.mean(dim=[2,3])  # [B, C]
#         std = x.std(dim=[2,3])    # [B, C]
#
#         style_feat = (mean * scale) + std * (1 - scale)
#         return style_feat  # [B, C]

class StyleExtractor(nn.Module):
    """
    提取风格特征，同时保留空间信息
    输入: style_image [B, C, H, W]
    输出: style_feat [B, C, H, W]
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        # 通道注意力
        avg_feat = self.global_avg_pool(x).view(B, C)
        scale = self.fc1(avg_feat)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale).view(B, C, 1, 1)  # 保留空间信息
        # 放大通道
        style_feat = x * scale
        return style_feat


class StyleAttention(nn.Module):
    """
    Q 来自生成图
    K, V = α * 风格 + (1-α) * 生成图
    """

    def __init__(self, in_channels):
        super().__init__()
        self.to_q = nn.Conv2d(in_channels, in_channels, 1)
        self.to_k = nn.Conv2d(in_channels, in_channels, 1)
        self.to_v = nn.Conv2d(in_channels, in_channels, 1)

        self.scale = (in_channels ** -0.5)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, q_feat, style_feat, alpha=0.5):
        """
        q_feat: 当前 UNet stage 的生成图特征   (B, C, H, W)
        style_feat: 对应 stage 的风格特征     (B, C, Hs, Ws)
        alpha: 风格注入比例
        """
        B, C, H, W = q_feat.shape

        # --- 自动调整风格特征分辨率 ---
        if style_feat.shape[2:] != (H, W):
            style_feat = F.interpolate(style_feat, size=(H, W), mode="bilinear", align_corners=False)

        # --- Q ---
        Q = self.to_q(q_feat)  # [B, C, H, W]

        # --- K/V 混合 ---
        K_gen = self.to_k(q_feat)
        K_style = self.to_k(style_feat)
        K = alpha * K_style + (1 - alpha) * K_gen

        V_gen = self.to_v(q_feat)
        V_style = self.to_v(style_feat)
        V = alpha * V_style + (1 - alpha) * V_gen

        # --- Flatten ---
        Q = Q.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        K = K.reshape(B, C, H * W)  # [B, C, HW]
        V = V.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        # --- 注意力 ---
        attn = torch.bmm(Q, K) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, V)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return q_feat + self.proj(out)  # 残差连接