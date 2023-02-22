import torch
import torch.nn as nn
from torch import einsum


class Attention(nn.Module):

    def __init__(self, params):
        super(Attention, self).__init__()

        self.params = params
        self.channel = params['encoder']['out_channels']
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']

        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.encoder_feature_conv = nn.Conv2d(self.channel, self.attention_dim, kernel_size=1)

        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

    def forward(self, cnn_features, hidden, alpha_sum, image_mask=None):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(einsum("b e h w->b h w e", alpha_sum_trans))  # 注意力分数转化为权重

        cnn_features_trans = self.encoder_feature_conv(cnn_features)

        alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + einsum("b e h w->b h w e", cnn_features_trans))
        energy = self.alpha_convert(alpha_score)
        # 下面的操作就是softmax操作，不直接使用softmax的原因是想使用mask屏蔽掉padding的影响
        energy = energy - energy.max()
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:, None, None] + 1e-10)

        alpha_sum = alpha[:, None, :, :] + alpha_sum

        context_vector = (alpha[:, None, :, :] * cnn_features).sum(-1).sum(-1)

        return context_vector, alpha, alpha_sum
