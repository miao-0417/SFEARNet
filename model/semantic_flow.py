import torch
import torch.nn as nn
from thop import profile
from torch.nn import functional as F

class Semantic_flow(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(Semantic_flow, self).__init__()
        #inchannel为低分辨率图片通道数，outchannel为高分辨率图片通道数
        self.down_h=nn.Conv2d( inchannel, outchannel,1,bias=False)
        self.down_l=nn.Conv2d( outchannel, outchannel,1,bias=False)
        self.flow_make=nn.Conv2d( outchannel*2,2,kernel_size=3,padding=1,bias=False)
        #self.conv=nn.Conv2d(inchannel*2,outchannel,1)
    def forward(self,h_feature, low_feature):
        low_feature, h_feature =low_feature, h_feature  # low_feature 对应分辨率较高的特征图，h_feature即为低分辨率的high-level feature
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        # 将high-level 和 low-level feature分别通过两个1x1卷积进行压缩
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        # 将high-level feature进行双线性上采样
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        # 预测语义流场 === 其实就是输入一个3x3的卷积
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        # 将Flow Field warp 到当前的 high-level feature中
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        #h_feature=self.conv(h_feature)
        return h_feature

    @staticmethod
    def flow_warp(inputs, flow, size):
        out_h, out_w = size  # 对应高分辨率的low-level feature的特征图尺寸
        n, c, h, w = inputs.size()  # 对应低分辨率的high-level feature的4个输入维度


        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)
        # 从-1到1等距离生成out_h个点，每一行重复out_w个点，最终生成(out_h, out_w)的像素点
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        # 生成w的转置矩阵
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        # 展开后进行合并
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        #print(grid.size())
        # grid指定由input空间维度归一化的采样像素位置，其大部分值应该在[ -1, 1]的范围内
        # 如x=-1,y=-1是input的左上角像素，x=1,y=1是input的右下角像素。
        # 具体可以参考《Spatial Transformer Networks》，下方参考文献[2]
        output = F.grid_sample(inputs, grid,align_corners=False)
        #print(output.size())
        return output


class Resampler(nn.Module):
    def __init__(self, input_size, output_size):
        super(Resampler, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, output_size), torch.linspace(-1, 1, output_size), indexing='ij')
        self.grid = torch.stack((grid_y, grid_x), 2).unsqueeze(0)

    def forward(self, input_tensor):
        grid = self.grid.repeat(input_tensor.size(0), 1, 1, 1).to(input_tensor.device)
        output_tensor = F.grid_sample(input_tensor, grid, align_corners=True)
        return output_tensor



if __name__=='__main__':
    low=torch.randn([8,128,16,16])
    high=torch.randn([8,256,8,8])
    model=Semantic_flow(256,128)
    model_x=model(high,low)
    print(model_x.size())

    x=torch.randn([1,8,64,64])
    model_2=Resampler(64,256)
    model2_x=model_2(x)
    print(model2_x.size())