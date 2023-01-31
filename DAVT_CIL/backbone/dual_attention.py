from math import ceil
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.functional import relu
# helpers
from torch.nn.utils import spectral_norm
from torch.nn import init
import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module

import sys
sys.path.append("/home/DAVT_CIL/backbone")

from modified_linear import CosineLinear
from backbone.myresnet import ResNet18Pre, ResNet164, ResNet18Pre224
from backbone.Resnet_cifar import resnet32
from backbone.Resnet_imageNet import resnet224



def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, l=3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))


def always(val):
    return lambda *args, **kwargs: val


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()  # dim=1 channel-wise
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout=0., SN=False):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(dim, dim * mult, 1)) if SN else nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Conv2d(dim * mult, dim, 1)) if SN else nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Dual_Attention_bn(nn.Module):
    def __init__(self, dim, fmap_size, heads=8, dim_key=32, dim_value=64, dropout=0., dim_out=None, downsample=False,
                 BN=True, SN=False):
        super().__init__()
        inner_dim_key = dim_key * heads
        inner_dim_value = dim_value * heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        ###########################
        self.inter_atten = None
        self.intra_atten = None
        ##########################

        if SN:
            self.to_q = nn.Sequential(
                spectral_norm(nn.Conv2d(dim, inner_dim_key, 1, stride=(2 if downsample else 1), bias=False)),
                nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_v = nn.Sequential(spectral_norm(nn.Conv2d(dim, inner_dim_value, 1, bias=False)),
                                      nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity())
            self.to_k = nn.Sequential(spectral_norm(nn.Conv2d(dim, inner_dim_key, 1, bias=False)),
                                      nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
        else:
            self.to_q = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, stride=(2 if downsample else 1), bias=False),
                                      nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_v = nn.Sequential(nn.Conv2d(dim, inner_dim_value, 1, bias=False),
                                      nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity())
            self.to_k = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, bias=False),
                                      nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())

        self.mk = nn.Sequential(
            nn.Sequential(nn.Conv2d(inner_dim_key, self.heads * fmap_size * fmap_size, 1, bias=False),
                          nn.BatchNorm2d(self.heads * fmap_size * fmap_size)),
        )

        self.attend = nn.Softmax(dim=-1)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        # self.external_k = nn.Sequential(
        #     nn.GELU(),
        #     spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1)) if SN else nn.Conv2d(inner_dim_value, dim_out, 1),
        #     out_batch_norm if BN else nn.Identity(),
        #     nn.Dropout(dropout)
        # )

        self.to_out = nn.Sequential(
            nn.GELU(),
            spectral_norm(nn.Conv2d(inner_dim_value * 2, dim_out, 1)) if SN else nn.Conv2d(inner_dim_value * 2, dim_out,
                                                                                       1),
            out_batch_norm if BN else nn.Identity(),
            nn.Dropout(dropout)
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step=(2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range), dim=-1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range), dim=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim=-1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q1 = self.to_q(x)
        y = q1.shape[2]

        # v = rearrange(self.to_v(x), 'b (h d) ... -> b h (...) d', h=h)
        qkv = (q1, self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=h), qkv)

        dots1 = self.mk(q1)

        dots1 = rearrange(dots1, 'b (h d) ... -> b h (...) d', h=h)

        dots2 = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots1 = self.apply_pos_bias(dots1)
        # dots2 = self.apply_pos_bias(dots2)

        attn1 = self.attend(dots1)
        # self.inter_atten = attn1.detach()  # 类间attention
        attn2 = self.attend(dots2)  # 类内attention

        # attn = torch.stack((attn2, attn1), dim=0)

        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S  效果不好，而且波动加大

        out1 = einsum('b h i j, b h j d -> b h i d', attn1, v)
        out1 = rearrange(out1, 'b h (x y) d -> b (h d) x y', h=h, y=y)

        out2 = einsum('b h i j, b h j d -> b h i d', attn2, v)
        out2 = rearrange(out2, 'b h (x y) d -> b (h d) x y', h=h, y=y)


        # out = (out1+out2) / 2
        self.inter_atten = out1.detach()  # 类间attention
        self.intra_atten = out2.detach()

        out = torch.cat((out1, out2), 1)

        # attn = (attn1 + attn2) / 2
        # self.fusion_atten = attn.detach()
        # out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # out = rearrange(out, 'b h (x y) d -> b (h d) x y', h=h, y=y)

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult=2, dropout=0., dim_out=None,
                 downsample=False, BN=True, SN=False, LN=False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.layers = nn.ModuleList([])
        self.attn_residual = (not downsample) and dim == dim_out

        ###########################
        self.inter_atten = None
        self.intra_atten = None
        ##########################

        if LN:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Dual_Attention_bn(dim, fmap_size=fmap_size, heads=heads, dim_key=dim_key,
                                                   dim_value=dim_value,
                                                   dropout=dropout, downsample=downsample, dim_out=dim_out, BN=BN,
                                                   SN=SN)),
                    PreNorm(dim_out, FeedForward(dim_out, mlp_mult, dropout=dropout, SN=SN))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Dual_Attention_bn(dim, fmap_size=fmap_size, heads=heads, dim_key=dim_key, dim_value=dim_value,
                                      dropout=dropout, downsample=downsample, dim_out=dim_out, BN=BN, SN=SN),
                    FeedForward(dim_out, mlp_mult, dropout=dropout, SN=SN)
                ]))

    def forward(self, x):
        ###########################
        self.inter_atten = None
        self.intra_atten = None
        ##########################
        for attn, ff in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x) + attn_res
            self.intra_atten = attn.intra_atten
            self.inter_atten = attn.inter_atten
            x = ff(x) + x
        return x


class DVT(nn.Module):
    def __init__(self, *,
                 image_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_mult,
                 stages=3,
                 dim_key=32,
                 dim_value=64,
                 dropout=0.,
                 cnnbackbone='ResNet32',
                 independent_classifier=False,
                 frozen_head=False,
                 BN=True,  # Batchnorm
                 LN=False,  # LayerNorm
                 SN=False,  # SpectralNorm
                 grow=False,  # Expand the network
                 mean_cob=False,
                 sum_cob=True,
                 max_cob=False,
                 distill_classifier=True,
                 cosine_classifier=False,
                 use_WA=False,
                 init="kaiming",
                 device='cuda',
                 use_bias=False, ):
        super(DVT, self).__init__()
        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        self.dims = dims
        self.depths = depths
        self.layer_heads = layer_heads
        self.image_size = image_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_mult = mlp_mult
        self.stages = stages
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dropout = dropout
        self.distill_classifier = distill_classifier
        self.cnnbackbone = cnnbackbone
        if image_size == 224 and num_classes == 100:
            self.cnnbackbone = 'ResNet18Pre224'  # ResNet18Pre224   PreActResNet
        self.nf = 64 if image_size < 100 else 32
        self.independent_classifier = independent_classifier
        self.frozen_head = frozen_head
        self.BN = BN
        self.SN = SN
        self.LN = LN
        self.grow = grow
        self.init = init
        self.use_WA = use_WA
        self.device = device
        self.weight_normalization = cosine_classifier
        self.use_bias = use_bias
        self.mean_cob = mean_cob
        self.sum_cob = sum_cob
        self.max_cob = max_cob
        self.gamma = None

        ###########################
        self.inter_atten = None
        self.intra_atten = None
        ##########################



        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), \
            'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        if self.cnnbackbone == 'ResNet18Pre':
            self.conv = ResNet18Pre(self.nf, self.stages)
        elif self.cnnbackbone == 'ResNet164':
            self.conv = ResNet164()
        elif self.cnnbackbone == 'ResNet18Pre224':
            self.conv = ResNet18Pre224(self.stages)
        elif self.cnnbackbone == 'resnet_cifar100':
            self.conv = resnet32()
        elif self.cnnbackbone == 'resnet_imageNet':
            self.conv = resnet224()
        else:
            assert ()

        if grow:
            print("Enable dynamical Transformer expansion!")
            self.transformers = nn.ModuleList()
            self.transformers.append(self.add_transformer())
        else:
            self.transformer = self.add_transformer()  # self.add_transformer()  # self.conv._resnet_high

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...')
        )
        # self.distill_head = self._gen_classifier(dims[-1], num_classes) if self.distill_classifier else always(None)
        #
        # # if self.independent_classifier:
        # #     task_class = 2 if num_classes < 20 else 20
        # #     self.fix = nn.ModuleList(
        # #         [self._gen_classifier(dims[-1], task_class) for i in range(num_classes // task_class)])
        # # else:
        # #     self.mlp_head = spectral_norm(self._gen_classifier(dims[-1], num_classes)) if SN else self._gen_classifier(
        # #         dims[-1], num_classes)
        #
        # self.feature_head = self._gen_classifier(dims[-1], num_classes)

        if self.stages == 1:
            self.fc = CosineLinear(dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.fc = CosineLinear(dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    def add_transformer(self):
        if self.nf == 64:
            fmap_size = self.image_size // ((2 ** 3) if self.stages < 3 else (2 ** 1))
        else:
            fmap_size = self.image_size // (2 ** 3)
        if self.cnnbackbone == 'ResNet18Pre224' or self.cnnbackbone == 'PreActResNet':
            if self.stages == 3:
                fmap_size = self.image_size // (2 ** 3)
            if self.stages == 2:
                fmap_size = self.image_size // (2 ** 4)
            if self.stages == 1:
                fmap_size = self.image_size // (2 ** 4)
        # elif self.cnnbackbone == 'resnet_cifar100':
        #     fmap_size = self.image_size // 4
        # elif self.cnnbackbone == 'resnet_imageNet':
        #     fmap_size = self.image_size // 4
        else:
            fmap_size = self.image_size // 4
        layers = []

        for ind, dim, depth, heads in zip(range(self.stages), self.dims, self.depths, self.layer_heads):
            is_last = ind == (self.stages - 1)
            layers.append(
                Transformer(dim, fmap_size, depth, heads, self.dim_key, self.dim_value, self.mlp_mult, self.dropout,
                            BN=self.BN, SN=self.SN, LN=self.LN))

            if not is_last:  # downsample
                next_dim = self.dims[ind + 1]
                layers.append(Transformer(dim, fmap_size, 1, heads * 2, self.dim_key, self.dim_value, dim_out=next_dim,
                                          downsample=True, BN=self.BN, SN=self.SN, LN=self.LN))
                fmap_size = ceil(fmap_size / 2)
        return nn.Sequential(*layers)

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineLinear(in_features, n_classes)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)
        return classifier

    # @torch.no_grad()
    # def update_gamma(self, task_num, class_per_task):
    #     if task_num == 0:
    #         return 1
    #     if self.distill_classifier:
    #         classifier = self.distill_head
    #     else:
    #         classifier = self.mlp_head
    #     old_weight_norm = torch.norm(classifier.weight[:task_num * class_per_task], p=2, dim=1)
    #     new_weight_norm = torch.norm(
    #         classifier.weight[task_num * class_per_task:task_num * class_per_task + class_per_task], p=2, dim=1)
    #     self.gamma = old_weight_norm.mean() / new_weight_norm.mean()
    #     print('gamma: ', self.gamma.cpu().item(), '  use_WA:', self.use_WA)
    #     if not self.use_WA:
    #         return 1
    #     return self.gamma

    def forward(self, img, return_feat=False):
        ###########################
        self.inter_atten = None
        self.intra_atten = None
        ##########################
        x = self.conv(img)
        if self.grow:
            x = [transformer(x) for transformer in self.transformers]
            if self.sum_cob:
                x = torch.stack(x).sum(dim=0)  # add growing transformers' output
            elif self.mean_cob:
                x = torch.stack(x).mean(dim=0)
            elif self.max_cob:
                for i in range(len(x) - 1):
                    x[i + 1] = x[i].max(x[i + 1])
                x = x[-1]
            else:
                ValueError
        else:
            x = self.transformer(x)
            self.inter_atten = self.transformer[-1].inter_atten
            self.intra_atten = self.transformer[-1].intra_atten

        x = self.pool(x)
        out = self.fc(x)

        # if self.independent_classifier:
        #     y = torch.tensor([])
        #     for fix in self.fix:
        #         y = torch.cat((fix(x), y), 1)
        #     out = y
        # else:
        #     out = self.mlp_head(x)

        # print('Out size:', out.size())
        if return_feat:
            return x, out
        return out


# 测试
if __name__ == '__main__':
    a = torch.randn((24, 3, 224, 224))
    model = DVT(image_size=224,
                num_classes=5,
                stages=3,  # number of stages
                dim=(128, 256, 512),  # dimensions at each stage
                depth=2,  # 2 transformer of depth 4 at each stage
                heads=(2, 4, 8),  # heads at each stage
                mlp_mult=2,  # 2
                cnnbackbone='ResNet18Pre224',  # 'ResNet18Pre' 'ResNet164'
                dropout=0.1)
    out = model(a)
    print(out.shape)
