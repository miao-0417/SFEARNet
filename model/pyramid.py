import torch
import torch.nn as nn
from thop import profile
from torch.nn import functional as F
from model.CBAM import CBAM
class Pyramid_Extraction(nn.Module):
    def __init__(self,channel, rate=1, bn_mom=0.1):
        super(Pyramid_Extraction, self).__init__()
        self.channel=channel

        self.branch1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=rate, dilation=rate, groups=channel,
                      bias=True),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel , kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(channel , momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel , momentum=bn_mom),
            nn.ReLU(inplace=True),
        )



        self.branch3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=4 * rate, dilation=4 * rate,
                      groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel , momentum=bn_mom),
            nn.ReLU(inplace=True),
        )



        self.branch4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=8 * rate, dilation=8 * rate,
                      groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel , kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel , momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch5_conv = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(channel, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)


        self.conv_cat = nn.Sequential(
            nn.Conv2d((channel ) * 5, (channel ) * 5, kernel_size=(1, 1), stride=(1, 1), padding=0,
                      groups=(channel ) * 5, bias=False),
            nn.BatchNorm2d((channel ) * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d((channel ) * 5, channel, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        [b,c,row,col]=x.size()

        conv1_1=self.branch1(x)
        #print(conv1_1.size())
        conv3_1=self.branch2(x)
        conv3_2=self.branch3(x)
        conv3_3=self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # 五个分支的内容堆叠起来，然后1x1卷积整合特征。
        feature_cat = torch.cat([conv1_1, conv3_1, conv3_2, conv3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result



class Pyramid_Merge(nn.Module):
    def __init__(self,channel):
        super(Pyramid_Merge, self).__init__()
        self.channel=channel
        self.cbam=CBAM(channel)
        self.pe=Pyramid_Extraction(channel)
        # self.conv=nn.Sequential(
        #     nn.Conv2d(channel*2,channel,kernel_size=1,stride=1),
        # )


        self.conv = nn.Sequential(
            nn.Conv2d(channel*2, channel*2, kernel_size=1, groups=channel*2),
            nn.BatchNorm2d(channel*2),
            nn.ReLU(),
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self,input1,input2):

        input_cat=torch.cat([input1,input2],dim=1)
        input_abs=torch.abs(input1-input2)

        #print(input_cat.size(),input_abs.size())
        input_cat_conv=self.conv(input_cat)
        #print(input_cat_conv.size())
        input_cat_conv_cbam=self.cbam(input_cat_conv)
        #print(input_cat_conv_cbam.size())
        # input_cat_cbam=self.cbam(input_cat)
        #input_cat_cbam_conv=self.conv(input_cat_cbam)

        input_abs_py=self.pe(input_abs)
        #print(input_abs_py.size())

        input_abs_py_abs=input_abs_py+input_abs

        out1=input_abs_py_abs+input_cat_conv_cbam

        return out1


if __name__=="__main__":
    model=Pyramid_Merge(256)
    x1=torch.randn([8,256,64,64])
    x2 = torch.randn([8, 256, 64, 64])
    model_x=model(x1,x2)
    print(model_x.size())



    input_data1 = torch.randn([1, 256, 64, 64])
    input_data2 = torch.randn([1, 256, 64, 64])
    flops, params = profile(model, inputs=(input_data1,input_data2,))
    print('Number of parameters: ' + str(params))
    print('FLOPs: ' + str(flops))
    print(params / 10 ** 6)
    print(flops / 10 ** 9)





