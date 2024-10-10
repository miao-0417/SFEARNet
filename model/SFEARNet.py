# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

#from model.backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from model.backbone import mit_b0, mit_b1

from model.semantic_flow import Semantic_flow,Resampler
from model.edge_aware import Edge_Guidance_1,Edge_Guidance_0
from model.pyramid import Pyramid_Merge
class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead_1(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead_1, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.sef1 = Semantic_flow(256, 64)
        self.sef2 = Semantic_flow(256, 64)
        self.sef3 = Semantic_flow(256, 64)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        #b0:torch.Size([8, 32, 64, 64]) torch.Size([8, 64, 32, 32]) torch.Size([8, 160, 16, 16]) torch.Size([8, 256, 8, 8])
        #print(c1.size(), c2.size(), c3.size(), c4.size())
        #b1 torch.Size([8, 64, 64, 64]) torch.Size([8, 128, 32, 32]) torch.Size([8, 320, 16, 16]) torch.Size([8, 512, 8, 8])
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        #print(_c4.size())
        _c4_1 = self.sef1(_c4, c1)
        # print(_c4.size())
        _c4_2 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # print(_c4.size())
        _c4 = _c4_1 + _c4_2

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])

        _c3_1 = self.sef2(_c3, c1)
        _c3_2 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # print(_c3.size())
        _c3 = _c3_1 + _c3_2

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2_1 = self.sef3(_c2, c1)
        _c2_2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = _c2_1 + _c2_2
        # print(_c2.size())

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        #print(x.size())
        return x


class SegFormerHead_0(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead_0, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.sef1 = Semantic_flow(256, 32)
        self.sef2 = Semantic_flow(256, 32)
        self.sef3 = Semantic_flow(256, 32)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # torch.Size([8, 32, 64, 64]) torch.Size([8, 64, 32, 32]) torch.Size([8, 160, 16, 16]) torch.Size([8, 256, 8, 8])
        # print(c1.size(), c2.size(), c3.size(), c4.size())
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4_1 = self.sef1(_c4, c1)
        #print(_c4.size())
        _c4_2 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # print(_c4.size())
        _c4 = _c4_1 + _c4_2

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])

        _c3_1 = self.sef2(_c3, c1)
        _c3_2 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # print(_c3.size())
        _c3 = _c3_1 + _c3_2

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        #print(_c2.size())
        _c2_1 = self.sef3(_c2, c1)
        _c2_2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = _c2_1 + _c2_2
        # print(_c2.size())

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print(x.size())
        return x
class SFEARNet(nn.Module):
    def __init__(self, num_classes=21, phi='b1', pretrained=False):
        super(SFEARNet, self).__init__()
        self.in_channels = {
            # 'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            # 'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            # 'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            # 'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
            'b0': mit_b0, 'b1': mit_b1,
        }[phi](pretrained)
        self.embedding_dim = {
            # 'b0': 256, 'b1': 256, 'b2': 768,
            # 'b3': 768, 'b4': 768, 'b5': 768,
            'b0': 256, 'b1': 256,
        }[phi]



        if phi=='b1':
            self.decode_head = SegFormerHead_1(num_classes, self.in_channels, self.embedding_dim)
            self.re = Resampler(64, 256)
            self.re2 = Resampler(64, 256)
            self.eg = Edge_Guidance_1()
            self.py1 = Pyramid_Merge(64)
            self.py2 = Pyramid_Merge(128)
            self.py3 = Pyramid_Merge(320)
            self.py4 = Pyramid_Merge(512)
        elif phi=='b0':
            self.decode_head = SegFormerHead_0(num_classes, self.in_channels, self.embedding_dim)
            self.re = Resampler(64, 256)
            self.re2 = Resampler(64, 256)
            self.eg = Edge_Guidance_0()
            self.py1 = Pyramid_Merge(32)
            self.py2 = Pyramid_Merge(64)
            self.py3 = Pyramid_Merge(160)
            self.py4 = Pyramid_Merge(256)

    def forward(self, input1,input2):
        H, W = input1.size(2), input1.size(3)

        x1 = self.backbone(input1)
        x2 = self.backbone(input2)
        #x=[torch.abs(xa-xb) for xa,xb in zip(x1,x2)]
        x_0=self.py1(x1[0],x2[0])
        x_1 = self.py2(x1[1],x2[1])
        x_2 = self.py3(x1[2], x2[2])
        x_3 = self.py4(x1[3], x2[3])
        x=[x_0,x_1,x_2,x_3]
        #print(x_3.size())

        eg=self.eg(x)
        edge=eg[0]
        x=eg[1:]
        #print(x[0].size())
        x = self.decode_head(x)

        #x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x=self.re(x)
        edge=self.re2(edge)
        return x,edge


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input1 = torch.rand([8, 3, 256, 256]).to(device)
    input2 = torch.rand([8, 3, 256, 256]).to(device)
    model = SFEARNet(num_classes=2, phi='b1',pretrained=False).to(device)
    out = model(input1,input2)
    print(out[0].size())
    print(out[1].size())

    input_data1 = torch.randn([1, 3, 256, 256]).to(device)
    input_data2 = torch.randn([1, 3, 256, 256]).to(device)
    flops, params = profile(model, inputs=(input_data1,input_data2,))
    print('Number of parameters: ' + str(params))
    print('FLOPs: ' + str(flops))
    print(params / 10 ** 6)
    print(flops / 10 ** 9)

# Number of parameters: 5563101.0
# FLOPs: 4645815552.0
# 5.563101
# 4.645815552

# Number of parameters: 20885663.0
# FLOPs: 14775373056.0
# 20.885663
# 14.775373056