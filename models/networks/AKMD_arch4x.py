# -*- coding: utf-8 -*-
"""
@Time ： 2023/12/26 20:45
@Auth ： Pengcheng Zheng
@File ：AKMD_arch4x.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
import torch.nn as nn
import math
import torch.nn.init as init
import os

from basicsr.archs.swinir_arch import DropPath
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
import pywt

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class _ResBLockDB(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBLockDB, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out


#################################------------------------SR_Branch-----------------------------------------------
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x

class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.MultiDomain = MultiDomainAttention(dim=64)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        # h = x.shape[2] // 2
        # w = int(h / 2) + 1
        res = self.residual_group(x)
        b_,c_,h_,w_ = res.shape
        res = res.reshape(b_,h_*w_,c_)
        res = self.MultiDomain(res)
        res = res.reshape(b_,c_,h_,w_)
        res = self.conv(res)
        return res + x

#[B,N,C]->[B,N,C]
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# MultiHeadSelfAttention [B,C,H,W]->[B,C,H,W]
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.query = torch.nn.Conv2d(embed_dim, 64, kernel_size=1)
        self.key = torch.nn.Conv2d(embed_dim, 64, kernel_size=1)
        self.value = torch.nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, self.num_heads, -1, self.head_dim)
        return x.permute(0, 1, 3, 2)

    def forward(self, x):
        batch_size, _, h, w = x.size()
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = self.softmax(scores)

        output = torch.matmul(attention_weights, value)
        output = output.permute(0, 1, 3, 2).contiguous().view(batch_size, self.num_heads, -1, h, w)
        output = output.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, -1, h, w)
        output = self.output_linear(output.view(batch_size,-1,_))
        output = output.view(batch_size, _, h, w)
        return output


# define wavelet transformm
def wavelet_transform(x):
    x = x.cpu().detach().numpy()  #tensor2ndarray
    coeffs = pywt.dwt2(x, 'haar')
    LL, (LH, HL, HH) = coeffs
    return torch.from_numpy(LL), torch.from_numpy(LH), torch.from_numpy(HL), torch.from_numpy(HH)

#[B,N,C]->[B,N,C]
class GlobalFilter(nn.Module):
    def __init__(self, dim, h=24, w=13):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02) #(12,7,64,2)
        #print(self.complex_weight.shape)
        self.w = w  #13
        self.h = h  #24

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape #4,144,64
        if spatial_size is None:
            a = b = int(math.sqrt(N)) #a=b=12
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C) #(4,12,12,64)

        x = x.to(torch.float32)
        #print(x.shape)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho') #(4,12,7,64)
        #print(x.shape)
        weight = torch.view_as_complex(self.complex_weight) #(12,7,64)
        #print(weight.shape)
        x = x * weight #(4,12,7,64)
        # print(x.shape)
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho') #(4,12,12,64)

        x = x.reshape(B, N, C) #(4,144,64)

        return x

class MultiDomainAttention(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,h=24,w=13):
        super(MultiDomainAttention, self).__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.conv1 = nn.Conv2d(192, 64, kernel_size=5, padding=5 // 2)
        self.multiheadattention = MultiHeadSelfAttention(embed_dim=dim, num_heads=4)
        self.convT2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

    def forward(self,x):
        raw = x #(4,24*24,64)
        x = self.norm1(x) #(4,576,64）
        b,n,c = x.shape #4,576,64
        x = x.reshape(b,c,int(math.sqrt(n)),int(math.sqrt(n))).permute(0, 1, 3, 2) #(4,64,24,24)
        LL, LH, HL, HH = wavelet_transform(x) #(4,64,12,12)
        high = self.conv1(torch.cat((LH,HL,HH),dim=1))  #(4,192,12,12)->(4,64,12,12)
        high_local_attention = self.multiheadattention(high) #(4,64,12,12)
        LL = LL.permute(0,2,3,1).contiguous().view(LL.size(0),-1,LL.size(1)) #(4,64,12,12)->(4,144,64)
        low_global_attention = self.filter(LL) #(4,144,64)
        low_global_attention = low_global_attention.reshape(b,c,
        int(math.sqrt(low_global_attention.shape[1])),int(math.sqrt(low_global_attention.shape[1]))).permute(0, 1, 3, 2) #(4,64,12,12)
        aligment = self.convT2(torch.cat((high_local_attention,low_global_attention),dim=1)) #(4,128,12,12)->(4,64,12,12)
        aligment = aligment.permute(0,2,3,1).contiguous().view(aligment.size(0),-1,aligment.size(1)) #(4,576,64)
        out = raw + self.drop_path(self.mlp(self.norm2(aligment)))
        return out

class SRBranch_0(nn.Module):
    """Residual Channel Attention Networks.

    ``Paper: Image Super-Resolution Using Very Deep Residual Channel Attention Networks``

    Reference: https://github.com/yulunzhang/RCAN

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=5,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(SRBranch_0, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        # self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.body(x)
        # res = self.conv_after_body(self.body(x))
        # res += x

        return res, x

class SRBranch_1(nn.Module):
    """Residual Channel Attention Networks.

    ``Paper: Image Super-Resolution Using Very Deep Residual Channel Attention Networks``

    Reference: https://github.com/yulunzhang/RCAN

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=5,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.):
        super(SRBranch_1, self).__init__()

        self.img_range = img_range
        # self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        # self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            ResidualGroup,
            num_group,
            num_feat=num_feat,
            num_block=num_block,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, res, x):
        # self.mean = self.mean.type_as(x)
        #
        # x = (x - self.mean) * self.img_range
        # x = self.conv_first(x)
        res = self.conv_after_body(self.body(res))
        res += x

        return res

#################-------------------------------------Deblur-Branch----------------------------------------------------

class SelectK(nn.Module):
    def __init__(self, input_size, inch):
        super(SelectK, self).__init__()
        self.input_size = input_size
        self.inch = inch
        self.layer0 = nn.Conv2d(self.inch, 1, kernel_size=1, stride=1, padding=0)
        self.fullconnect = nn.Sequential(
            nn.Linear(self.input_size * self.input_size, self.input_size * self.input_size * 3 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size * self.input_size * 3 // 4, self.input_size * self.input_size),
            nn.ReLU(inplace=True),
        )
        self.min_value = (torch.sin(nn.Parameter(torch.rand(1))) / 2)
        self.max_value = (self.min_value + 0.5)

    def selectMap(self, input, min_threadhold, max_threadhold):
        map_0 = torch.zeros_like(input)
        map_1 = torch.zeros_like(input)
        map_2 = torch.zeros_like(input)
        # 将小于最小阈值的元素映射为3
        map_0[input < min_threadhold] = 1
        # 将大于最大阈值的元素映射为7
        map_2[(input > max_threadhold)] = 1
        # 其余元素映射为5
        map_1[(input >= min_threadhold) & (input <= max_threadhold)] = 1
        return (map_0 - input).detach() + input, (map_1 - input).detach() + input, (map_2 - input).detach() + input

    def forward(self, x):
        out = x

        out = self.layer0(out)

        out = out.view(out.shape[0], 1, self.input_size * self.input_size)

        out = self.fullconnect(out)

        out = torch.sigmoid(out)

        out = out.view(out.shape[0], 1, self.input_size, self.input_size)

        return self.selectMap(out, 0.2, 0.8)


class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            self.activation,
        )
        self.en_layer1_2_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=1, padding=0))
        self.en_layer1_2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.en_layer1_2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=5, padding=2))

        self.en_layer1_3_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=1, padding=0))
        self.en_layer1_3_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.en_layer1_3_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=5, padding=2))

        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        self.en_layer2_2_0 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=1, padding=0))
        self.en_layer2_2_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.en_layer2_2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=5, padding=2))

        self.en_layer2_3_0 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            self.activation)
        self.en_layer2_3_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation)
        self.en_layer2_3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            self.activation)

        # self.en_layer3_1 = nn.Sequential(
        #     nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1),
        #     self.activation,
        # )

    def forward(self, x):
        hx = self.en_layer1_1(x)
        selectK1 = SelectK(hx.shape[-1], 64)
        print("en1:",hx.shape)
        map_0, map_1, map_2 = selectK1(hx)
        hx = self.activation(
            self.en_layer1_2_0(hx) * map_0 + self.en_layer1_2_1(hx) * map_1 + self.en_layer1_2_2(hx) * map_2 + hx)
        hx = self.activation(
            self.en_layer1_3_0(hx) * map_0 + self.en_layer1_3_1(hx) * map_1 + self.en_layer1_3_2(hx) * map_2 + hx)
        residual_1 = hx

        hx = self.en_layer2_1(hx)
        selectK2 = SelectK(hx.shape[-1], 128)
        print("en2:", hx.shape)
        map_0, map_1, map_2 = selectK2(hx)
        hx = self.activation(
            self.en_layer2_2_0(hx) * map_0 + self.en_layer2_2_1(hx) * map_1 + self.en_layer2_2_2(hx) * map_2 + hx)
        hx = self.activation(
            self.en_layer2_3_0(hx) * map_0 + self.en_layer2_3_1(hx) * map_1 + self.en_layer2_3_2(hx) * map_2 + hx)
        residual_2 = hx

        # hx = self.en_layer3_1(hx)

        return hx, residual_1, residual_2

class Embeddings_output(nn.Module):
    def __init__(self):
        super(Embeddings_output, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        # DStage1####################
        self.de_layer3_0 = nn.Sequential(
            nn.ConvTranspose2d(320, 192, kernel_size=4, stride=2, padding=1),
            self.activation,
        )

        self.de_layer2_0 = nn.Sequential(
            nn.Conv2d(192 + 128, 192, kernel_size=1, padding=0),
            self.activation,
        )
        #第一个PAKB
        self.de_layer2_1_0 = nn.Sequential(
            nn.Conv2d(192,192,kernel_size=1,padding=0),
            self.activation,
            nn.Conv2d(192, 192, kernel_size=1, padding=0),
        )
        self.de_layer2_1_1 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
        )
        self.de_layer2_1_2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=5, padding=2),
            self.activation,
            nn.Conv2d(192, 192, kernel_size=5, padding=2),
        )
        #第二个PAKB
        self.de_layer2_2_0 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=1, padding=0),
            self.activation,
        )
        self.de_layer2_2_1 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            self.activation,
        )
        self.de_layer2_2_2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=5, padding=2),
            self.activation,
        )

        #DStage2####################
        self.de_layer1_0 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1),
            self.activation,
        )

        self.de_layer1_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
            self.activation,
        )
        self.de_layer1_1_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
        )
        self.de_layer1_1_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.de_layer1_1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        )
        self.de_layer1_2_0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            self.activation,
        )
        self.de_layer1_2_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
        )
        self.de_layer1_2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            self.activation,
        )
        self.de_layer1_3 = nn.Sequential(
            nn.Conv2d(64,3,kernel_size=3,padding=1),
            self.activation
        )

    def forward(self, x, residual_1, residual_2):
        hx = self.de_layer3_0(x)
        hx = self.de_layer2_0(torch.cat((hx, residual_2), dim=1))

        selectK1 = SelectK(hx.shape[-1], 192)
        map_0, map_1, map_2 = selectK1(hx)
        hx = self.activation(
            self.de_layer2_1_0(hx) * map_0 + self.de_layer2_1_1(hx) * map_1 + self.de_layer2_1_2(hx) * map_2 + hx)
        hx = self.activation(
            self.de_layer2_2_0(hx) * map_0 + self.de_layer2_2_1(hx) * map_1 + self.de_layer2_2_2(hx) * map_2 + hx)


        hx = self.de_layer1_0(hx)
        hx = self.de_layer1_1(torch.cat((hx, residual_1), dim=1))

        selectK2 = SelectK(hx.shape[-1], 64)
        map_0, map_1, map_2 = selectK2(hx)
        hx = self.activation(
            self.de_layer1_1_0(hx) * map_0 + self.de_layer1_1_1(hx) * map_1 + self.de_layer1_1_2(hx) * map_2 + hx)
        hx = self.activation(
            self.de_layer1_2_0(hx) * map_0 + self.de_layer1_2_1(hx) * map_1 + self.de_layer1_2_2(hx) * map_2 + hx)

        hx = self.de_layer1_3(hx)

        return hx


class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        B, N, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out


class Mlp_(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp_, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


# CPE (Conditional Positional Embedding)
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x


class Intra_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Intra_SA, self).__init__()
        self.hidden_size = dim // 2
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_h
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_v
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp_(dim)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
        feature_h = feature_h.view(B * H, W, C // 2)
        feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
        feature_v = feature_v.view(B * W, H, C // 2)
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
        qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]

        if H == W:
            query = torch.cat((q_h, q_v), dim=0)
            key = torch.cat((k_h, k_v), dim=0)
            value = torch.cat((v_h, v_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, W, C // 2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C // 2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(q_h, k_h, v_h)
            attention_output_v = self.attn(q_v, k_v, v_v)
            attention_output_h = attention_output_h.view(B, H, W, C // 2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C // 2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x


class Inter_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Inter_SA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size // 2, 3 * (self.hidden_size // 2), kernel_size=1, padding=0)  # qkv_h
        self.conv_v = nn.Conv2d(self.hidden_size // 2, 3 * (self.hidden_size // 2), kernel_size=1, padding=0)  # qkv_v
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = Mlp_(self.hidden_size)
        self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = torch.chunk(self.conv_h(x_input[0]), 3, dim=1)
        feature_v = torch.chunk(self.conv_v(x_input[1]), 3, dim=1)
        query_h, key_h, value_h = feature_h[0], feature_h[1], feature_h[2]
        query_v, key_v, value_v = feature_v[0], feature_v[1], feature_v[2]

        horizontal_groups = torch.cat((query_h, key_h, value_h), dim=0)
        horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous()
        horizontal_groups = horizontal_groups.view(3 * B, H, -1)
        horizontal_groups = torch.chunk(horizontal_groups, 3, dim=0)
        query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]

        vertical_groups = torch.cat((query_v, key_v, value_v), dim=0)
        vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()
        vertical_groups = vertical_groups.view(3 * B, W, -1)
        vertical_groups = torch.chunk(vertical_groups, 3, dim=0)
        query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]

        if H == W:
            query = torch.cat((query_h, query_v), dim=0)
            key = torch.cat((key_h, key_v), dim=0)
            value = torch.cat((value_h, value_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, C // 2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C // 2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(query_h, key_h, value_h)
            attention_output_v = self.attn(query_v, key_v, value_v)
            attention_output_h = attention_output_h.view(B, H, C // 2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C // 2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x


class Deblur_branch_0(nn.Module):
    def __init__(self):
        super(Deblur_branch_0, self).__init__()

        self.encoder = Embeddings()

    def forward(self, x):
        hx, residual_1, residual_2 = self.encoder(x)

        return hx, residual_1, residual_2


class Deblur_branch_1(nn.Module):
    def __init__(self):
        super(Deblur_branch_1, self).__init__()

        # self.encoder = Embeddings()
        self.activation = nn.LeakyReLU(0.2, True)
        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        head_num = 5
        dim = 320
        self.Trans_block_1 = Intra_SA(dim, head_num)
        self.Trans_block_2 = Inter_SA(dim, head_num)
        self.Trans_block_3 = Intra_SA(dim, head_num)
        self.Trans_block_4 = Inter_SA(dim, head_num)
        self.Trans_block_5 = Intra_SA(dim, head_num)
        self.Trans_block_6 = Inter_SA(dim, head_num)
        self.Trans_block_7 = Intra_SA(dim, head_num)
        self.Trans_block_8 = Inter_SA(dim, head_num)
        self.Trans_block_9 = Intra_SA(dim, head_num)
        self.Trans_block_10 = Inter_SA(dim, head_num)
        self.Trans_block_11 = Intra_SA(dim, head_num)
        self.Trans_block_12 = Inter_SA(dim, head_num)
        self.decoder = Embeddings_output()

    def forward(self, x, residual_1, residual_2, x_0):
        # hx, residual_1, residual_2 = self.encoder(x)
        hx = self.en_layer3_1(x)

        hx = self.Trans_block_1(hx)
        hx = self.Trans_block_2(hx)
        hx = self.Trans_block_3(hx)
        hx = self.Trans_block_4(hx)
        hx = self.Trans_block_5(hx)
        hx = self.Trans_block_6(hx)

        hx = self.Trans_block_7(hx)
        hx = self.Trans_block_8(hx)
        hx = self.Trans_block_9(hx)
        hx = self.Trans_block_10(hx)
        hx = self.Trans_block_11(hx)
        hx = self.Trans_block_12(hx)
        hx = self.decoder(hx, residual_1, residual_2)
        deblur_feature = hx + x_0
        deblur_out = deblur_feature

        return deblur_feature, deblur_out


#################################-----------------------------------------------------------------------------------

class _GateMoudle(nn.Module):
    def __init__(self):
        super(_GateMoudle, self).__init__()

        self.conv1 = nn.Conv2d(131, 64, (3, 3), 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, (1, 1), 1, padding=0)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        scoremap = self.conv2(con1)
        return scoremap

class _ReconstructMoudle(nn.Module):
    def __init__(self):
        super(_ReconstructMoudle, self).__init__()
        self.resBlock = self._makelayers(64, 64, 8)
        self.conv1 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle1 = nn.PixelShuffle(2)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 256, (3, 3), 1, 1)
        self.pixelShuffle2 = nn.PixelShuffle(2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1, 1)
        # self.conv3 = nn.Conv2d(16, 16, (3, 3), 1, 1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(64, 3, (3, 3), 1, 1)
        # self.conv4 = nn.Conv2d(16, 3, (3, 3), 1, 1)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        res1 = self.resBlock(x)
        con1 = self.conv1(res1) #(4,256,24,24)
        pixelshuffle1 = self.relu1(self.pixelShuffle1(con1)) #(4,64,48,48)

        con2 = self.conv2(pixelshuffle1) #(4,256,48,48)
        pixelshuffle2 = self.relu2(self.pixelShuffle2(con2))

        con3 = self.relu3(self.conv3(pixelshuffle2)) #(4,64,48,48)
        print("con3",con3.shape)
        sr_deblur = self.conv4(con3) #(4,3,48,48)
        return sr_deblur

class FeatExchange(nn.Module):
    def __init__(self):
        super(FeatExchange, self).__init__()
        self.up = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.down = nn.Conv2d(64, 128, kernel_size=3, stride = 2, padding=1)
        self.alpha = nn.Parameter(torch.rand(1))
    def forward(self, deblur_feat, sr_feat):
        feat0 = self.up(deblur_feat)
        feat1 = self.down(sr_feat)
        deblur_feat = deblur_feat*self.alpha + (1-self.alpha)*feat1
        sr_feat = sr_feat*self.alpha + (1-self.alpha)*feat0
        return deblur_feat, sr_feat

# @ARCH_REGISTRY.register()
class AKMF_Net(nn.Module):
    def __init__(self):
        super(AKMF_Net, self).__init__()
        self.deblurBranch_0 = Deblur_branch_0()
        self.deblurBranch_1 = Deblur_branch_1()
        self.conv = nn.Conv2d(3, 64, kernel_size=5, padding=5 // 2)
        self.srBranch_0 = SRBranch_0(num_in_ch=3, num_out_ch=3, num_feat=64, num_group=5, num_block=5,
                                     squeeze_factor=16, upscale=4, res_scale=1, img_range=255.,
                                     rgb_mean=[0.4488, 0.4371, 0.4040])
        self.srBranch_1 = SRBranch_1(num_in_ch=3, num_out_ch=3, num_feat=64, num_group=5, num_block=5,
                                     squeeze_factor=16, upscale=4, res_scale=1, img_range=255., )
        self.featExchange = FeatExchange()
        self.geteMoudle = self._make_net(_GateMoudle)
        self.reconstructMoudle = self._make_net(_ReconstructMoudle)

    def forward(self, x, gated=True, isTest=True):
        if isTest == True:
            origin_size = x.size()
            input_size = (math.ceil(origin_size[2] / 2) * 2, math.ceil(origin_size[3] / 2) * 2)
            out_size = (origin_size[2] * 2, origin_size[3] * 2)
            x = nn.functional.upsample(x, size=input_size, mode='bilinear')

        deblur_feat0, residual_1, residual_2 = self.deblurBranch_0(x)
        sr_feat0, sr_x = self.srBranch_0(x)
        deblur_feat0, sr_feat0 = self.featExchange(deblur_feat0, sr_feat0)

        deblur_feature, deblur_out = self.deblurBranch_1(deblur_feat0, residual_1, residual_2, x)
        deblur_feature = self.conv(deblur_feature)  # (4,64,24,24)

        sr_feature = self.srBranch_1(sr_feat0, sr_x)

        if gated == True:
            scoremap = self.geteMoudle(torch.cat((deblur_feature, x, sr_feature), 1))  # (4,131,24,24)->(4,64,24,24)
        else:
            scoremap = torch.cuda.FloatTensor().resize_(sr_feature.shape).zero_() + 1
        repair_feature = torch.mul(scoremap, deblur_feature)  # (4,64,24,24)*(4,64,24,24)
        fusion_feature = torch.add(sr_feature, repair_feature)  # (4,64,24,24)+(4,64,24,24)
        recon_out = self.reconstructMoudle(fusion_feature)

        # if isTest == True:
        #     recon_out = nn.functional.upsample(recon_out, size=out_size, mode='bilinear')

        return deblur_out, recon_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)

if __name__=='__main__':
    model = AKMF_Net()
    print(model)
    x= torch.Tensor(1,3,48,48)
    lr_deblur, sr = model(x)
    print("lr_deblur", lr_deblur.shape)
    print("sr", sr.shape)
