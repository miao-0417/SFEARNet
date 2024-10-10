import torch
import torch.nn as nn
from thop import profile
from torch.nn import functional as F
from model.semantic_flow import Semantic_flow
from model.CBAM import CBAM
class Doubleconv(nn.Module):
    def __init__(self, input_channels, num_channels,):
        super().__init__()
        # self.conv1 = nn.Conv2d(input_channels, num_channels,
        #                        kernel_size=3, padding=1, stride=strides),
        # self.conv1=nn.Sequential(
        #     nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides),
        #     nn.BatchNorm2d(num_channels),
        #     nn.ReLU()
        #
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # self.conv2 = nn.Conv2d(num_channels, num_channels,
        #                        kernel_size=3, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

    def forward(self, X):
        Y = self.conv1(X)
        Y = self.conv2(Y)
        return Y

class Edge_Aware_1(nn.Module):
    def __init__(self,channel1,channel2):
        super(Edge_Aware_1, self).__init__()
        # self.conv_channel1=nn.Sequential(
        #     nn.Conv2d(channel1,channel1//4,kernel_size=1),
        #     nn.BatchNorm2d(channel1//4),
        #     nn.ReLU
        # )
        self.conv_channel1 = nn.Sequential(
            nn.Conv2d(channel1, channel1, kernel_size=3, padding=1, groups=channel1),
            nn.BatchNorm2d(channel1),
            nn.ReLU(),
            nn.Conv2d(channel1, channel1, kernel_size=1),
            nn.BatchNorm2d(channel1 ),
            nn.ReLU()
        )
        self.conv_channel2 = nn.Sequential(
            nn.Conv2d(channel2, channel2, kernel_size=3, padding=1, groups=channel2),
            nn.BatchNorm2d(channel2),
            nn.ReLU(),
            nn.Conv2d(channel2, channel2, kernel_size=1),
            nn.BatchNorm2d(channel2 ),
            nn.ReLU()
        )

        self.doubleconv=Doubleconv(channel1 +channel2,(channel1 +channel2)*2)
        self.conv1=nn.Sequential(
            nn.Conv2d((channel1 +channel2)*2,2,kernel_size=1),
            nn.BatchNorm2d(2),
            #nn.Sigmoid()
        )
        self.sef = Semantic_flow(512, 64)

    def forward(self,input1,input4):
        #print(input1.size(),input4.size())
        input1=self.conv_channel1(input1)
        input4=self.conv_channel2(input4)
        #print(input1.size(),input4.size())
        input4_sef=self.sef(input4,input1)
        input4_big=F.interpolate(input4, size=input1.size()[2:], mode='bilinear', align_corners=False)+input4_sef
        input=torch.cat([input4_big,input1],dim=1)
        input_res=self.doubleconv(input)
        out=self.conv1(input_res)
        return out
class Edge_Aware_0(nn.Module):
    def __init__(self,channel1,channel2):
        super(Edge_Aware_0, self).__init__()
        # self.conv_channel1=nn.Sequential(
        #     nn.Conv2d(channel1,channel1//4,kernel_size=1),
        #     nn.BatchNorm2d(channel1//4),
        #     nn.ReLU
        # )
        self.conv_channel1 = nn.Sequential(
            nn.Conv2d(channel1, channel1, kernel_size=3, padding=1, groups=channel1),
            nn.BatchNorm2d(channel1),
            nn.ReLU(),
            nn.Conv2d(channel1, channel1 , kernel_size=1),
            nn.BatchNorm2d(channel1 ),
            nn.ReLU()
        )
        self.conv_channel2 = nn.Sequential(
            nn.Conv2d(channel2, channel2, kernel_size=3, padding=1, groups=channel2),
            nn.BatchNorm2d(channel2),
            nn.ReLU(),
            nn.Conv2d(channel2, channel2, kernel_size=1),
            nn.BatchNorm2d(channel2 ),
            nn.ReLU()
        )

        self.doubleconv=Doubleconv(channel1 +channel2,(channel1 +channel2)*2)
        self.conv1=nn.Sequential(
            nn.Conv2d((channel1 +channel2)*2,2,kernel_size=1),
            nn.BatchNorm2d(2),
            #nn.Sigmoid()
        )

        self.sef = Semantic_flow(256, 32)

    def forward(self,input1,input4):
        input1=self.conv_channel1(input1)
        input4=self.conv_channel2(input4)
        #print(input1.size(),input4.size())
        input4_sef=self.sef(input4,input1)
        input4_big=F.interpolate(input4, size=input1.size()[2:], mode='bilinear', align_corners=False)+input4_sef
        #print(input4_big.size())
        input=torch.cat([input4_big,input1],dim=1)
        input_res=self.doubleconv(input)
        out=self.conv1(input_res)
        return out
class Edge_Guidance_1(nn.Module):
    def __init__(self):
        super(Edge_Guidance_1, self).__init__()

        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.ea=Edge_Aware_1(64,512)
        self.conv2_1=nn.Sequential(
            nn.Conv2d(2,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.ca1=CBAM(64)
        self.ca2=CBAM(128)
        self.ca3 = CBAM(320)
        self.ca4 = CBAM(512)

    def forward(self,input):
        input1,input2,input3,input4=input
        edge_64_2=self.ea(input1,input4)
        edge_64=self.conv2_1(edge_64_2)
        #print(edge_64_1.size())
        edge_32=self.maxpool1(edge_64)
        edge_16=self.maxpool2(edge_32)
        edge_8=self.maxpool3(edge_16)


        feature_1=input1*edge_64+input1
        feature_2=input2*edge_32+input2
        feature_3=input3*edge_16+input3
        feature_4=input4*edge_8+input4
        #print(feature_1.size(),feature_2.size(),feature_3.size(),feature_4.size())

        feature_1=self.ca1(feature_1)
        feature_2 = self.ca2(feature_2)
        feature_3 = self.ca3(feature_3)
        feature_4 = self.ca4(feature_4)

        return edge_64_2,feature_1,feature_2,feature_3,feature_4

class Edge_Guidance_0(nn.Module):
    def __init__(self):
        super(Edge_Guidance_0, self).__init__()

        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.ea=Edge_Aware_0(32,256)
        self.conv2_1=nn.Sequential(
            nn.Conv2d(2,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.ca1=CBAM(32)
        self.ca2=CBAM(64)
        self.ca3 = CBAM(160)
        self.ca4 = CBAM(256)

    def forward(self,input):
        input1,input2,input3,input4=input
        edge_64_2=self.ea(input1,input4)
        edge_64=self.conv2_1(edge_64_2)
        #print(edge_64_1.size())
        edge_32=self.maxpool1(edge_64)
        edge_16=self.maxpool2(edge_32)
        edge_8=self.maxpool3(edge_16)

        feature_1=input1*edge_64+input1
        feature_2=input2*edge_32+input2
        feature_3=input3*edge_16+input3
        feature_4=input4*edge_8+input4

        feature_1=self.ca1(feature_1)
        feature_2 = self.ca2(feature_2)
        feature_3 = self.ca3(feature_3)
        feature_4 = self.ca4(feature_4)

        return edge_64_2,feature_1,feature_2,feature_3,feature_4












