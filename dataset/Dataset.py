 #%matplotlib inline
import torchvision
import torch
from torchvision import transforms
import os
import glob
from PIL import Image
from osgeo import gdal
from train_options import parser
from torch.utils.data import DataLoader
from dataset.dataAug import RandomHorizontalFlip,RandomVerticalFlip,RandomFixRotate,RandomExchangeOrder
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import cv2


augment_transforms = transforms.Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomFixRotate(),
    RandomExchangeOrder()])

class Dataset(torch.utils.data.Dataset):
    def __init__(self,imge_dir_A,imge_dir_B,label_dir,edge_dir,is_train=True):
        self.ids=os.listdir(imge_dir_A)
        self.images_fps_A=[os.path.join(imge_dir_A,image_id) for image_id in self.ids]
        self.images_fps_B = [os.path.join(imge_dir_B, image_id) for image_id in self.ids]
        self.label_fps = [os.path.join(label_dir, image_id) for image_id in self.ids]
        self.edge_fps = [os.path.join(edge_dir, image_id) for image_id in self.ids]
        self.is_train=is_train
    def __getitem__(self, item):
        image_item_A=self.images_fps_A[item]
        image_item_B = self.images_fps_B[item]
        label_item = self.label_fps[item]
        edge_item = self.edge_fps[item]

        image_A=cv2.imread(image_item_A)
        image_B=cv2.imread(image_item_B)
        label=cv2.imread(label_item,cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_item, cv2.IMREAD_GRAYSCALE)
        img_ids = self.ids[item]

        if self.is_train:
            img1, img2, label,edge = augment_transforms([image_A, image_B, label,edge])
            label = label // 255
            label_tensor = torch.tensor(label).type(torch.long)

            edge = edge // 255
            edge_tensor = torch.tensor(edge).type(torch.long)
            # img1=img1.transpose(2, 0, 1)
            # img2 = img2.transpose(2, 0, 1)
            #
            # image_tensor_A = torch.tensor(img1).type(torch.float)
            # image_tensor_B=torch.tensor(img2).type(torch.float)
            image_tensor_A = TF.to_tensor(img1).type(torch.float)
            image_tensor_B = TF.to_tensor(img2).type(torch.float)
            #label_tensor= TF.to_tensor(label).type(torch.long)
            #print(image_tensor_A.size(), image_tensor_B.size(), label_tensor.size(),)
            #label_tensor = torch.squeeze(label_tensor)
        else:
            label = label // 255
            edge=edge//255
            # img1 = image_A.transpose(2, 0, 1)
            # img2 = image_B.transpose(2, 0, 1)
            #
            # image_tensor_A = torch.tensor(img1).type(torch.float)
            # image_tensor_B = torch.tensor(img2).type(torch.float)
            image_tensor_A = TF.to_tensor(image_A).type(torch.float)
            image_tensor_B = TF.to_tensor(image_B).type(torch.float)
            label_tensor = torch.tensor(label).type(torch.long)
            edge_tensor = torch.tensor(edge).type(torch.long)
            # label_tensor = torch.squeeze(label_tensor)
        return image_tensor_A, image_tensor_B, label_tensor, edge_tensor,img_ids

    def __len__(self):
        return len(self.ids)

