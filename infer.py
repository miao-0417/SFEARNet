# coding=utf-8
import os
import datetime

import pandas as pd
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from model.SFEARNet import SFEARNet
from dataset.Dataset import Dataset
from train_options import parser
from Metric import SegmentationMetric

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###############需要改的部分

save_dir = './vis/gz/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

n_classes=2

re=[]
if __name__ == '__main__':
    ###############加载数据
    DATA_DIR = opt.data_dir  # 根据自己的路径来设置
    # print(DATA_DIR)
    test_dir_A = os.path.join(DATA_DIR, 'test/A')
    test_dir_B = os.path.join(DATA_DIR, 'test/B')
    test_dir_label = os.path.join(DATA_DIR, 'test/label')
    test_dir_edge = os.path.join(DATA_DIR, 'test/edge')


    test_dataset = Dataset(test_dir_A, test_dir_B, test_dir_label,test_dir_edge,is_train=False)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = torch.load('GZ_CD_result/epoch/SFEARNet-0.0001-8-0.001-1-100/netCD_epoch.pth').to(device)

    model.eval()
    test_SegmentationMetric = SegmentationMetric(numClass=n_classes)
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for  image_A, image_B, label,edge,img_id in test_bar:
            image_A, image_B, label,edge = image_A.to(device), image_B.to(device), label.to(device),edge.to(device)
            img_id_str=",".join(img_id)
            #print(img_id_str)
            #break
            model_x = model(image_A, image_B)
            prob=model_x[0]
            label_pred1 = torch.argmax(prob, dim=1)
            #print(label_pred1.size())

            test_confusion=test_SegmentationMetric.addBatch(label_pred1,label)

            file_path = save_dir + img_id_str.split('.')[-2].split('\\')[-1] + '.png'

            cd_pred1 = label_pred1.unsqueeze(1)

            cd_preds = label_pred1.data.cpu().numpy()
            #cd_preds = cd_preds.swapaxes(0, 1).squeeze() * 255
            cd_preds = cd_preds.squeeze() * 255

            cv2.imwrite(file_path, cd_preds)

        test_PA = test_SegmentationMetric.PixelAccuary()
        test_OA = test_SegmentationMetric.OverallAccuary()
        test_IOU = test_SegmentationMetric.IntersectionOverUnion()
        test_recall = test_SegmentationMetric.recall()
        test_F1 = test_SegmentationMetric.F1()
        test_kappa=test_SegmentationMetric.kappa()
        print('OA:', format(test_OA, ".4f"), 'PA:',format(test_PA, ".4f"), 'recall:',format(test_recall, ".4f"), 'f1:',format(test_F1, ".4f"), 'iou:',format(test_IOU, ".4f"),  'kappa:',format(test_kappa, ".4f"),)
        re.append([test_OA,test_PA,test_recall,test_F1,test_IOU,test_kappa])

        data = pd.DataFrame(data=re, index=None, columns=['OA', 'PA', 'recall', 'f1', 'iou', 'kappa'])
        # #print(data)
        data.to_csv(save_dir+'infer.csv')








