from tqdm import tqdm
from dataset.Dataset import Dataset

from train_options import parser
import os
from torch.utils.data import DataLoader
import pandas as pd

from loss.Diceloss import Diceloss
from loss.Iouloss import IoULoss
import torch
from model.SFEARNet import SFEARNet
from torch.optim import lr_scheduler
import random
import numpy as np
from tensorboardX import SummaryWriter
from Metric import SegmentationMetric

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
opt = parser.parse_args()

# set seeds
seed_torch(0)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_decay=opt.weight_decay
lamda=opt.lamda
model = SFEARNet(2, phi='b0',pretrained=True).to(device, dtype=torch.float)


optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr,weight_decay=weight_decay)
lr_scheduler_model=lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience = 6,
        min_lr = 1e-20,
        verbose=True
    )

criterion_iou=IoULoss()
# weight = torch.tensor([2])  # 设置权重
# criterion_weighted_ce = CELoss_Weighted(weight=weight)
criterion_ce=torch.nn.CrossEntropyLoss()
criterion_dice_edge = Diceloss()


writer = SummaryWriter(opt.result_dir+"log/" + f'{opt.model}-{opt.lr}-{opt.train_batchsize}-{opt.weight_decay}-{opt.lamda}-{opt.num_epochs}/')
save_model_dir=opt.result_dir+"epoch/" + f'{opt.model}-{opt.lr}-{opt.train_batchsize}-{opt.weight_decay}-{opt.lamda}-{opt.num_epochs}/'
save_csv_dir=opt.result_dir+"csv/" + f'{opt.model}-{opt.lr}-{opt.train_batchsize}-{opt.weight_decay}-{opt.lamda}-{opt.num_epochs}/'
data_dir=opt.data_dir
Batch_Size=opt.train_batchsize
train_dir_A = os.path.join(data_dir, 'train/A/')
train_dir_B=os.path.join(data_dir, 'train/B/')
train_label_dir = os.path.join(data_dir, 'train/label/')
train_edge_dir = os.path.join(data_dir, 'train/edge/')

val_dir_A = os.path.join(data_dir, 'val/A/')
val_dir_B = os.path.join(data_dir, 'val/B/')
val_label_dir = os.path.join(data_dir, 'val/label/')
val_edge_dir = os.path.join(data_dir, 'val/edge/')
max_IOU =0.5

if __name__ == '__main__':
    train_dataset = Dataset(train_dir_A, train_dir_B, train_label_dir,train_edge_dir,is_train=True)
    val_dataset = Dataset(val_dir_A, val_dir_B, val_label_dir,val_edge_dir,is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
    result_train=[]
    result_val=[]

    for epoch_id in range(1, opt.num_epochs + 1):
        #print()
        #train_running_results = {'loss': 0, 'PA': 0, 'mPA': 0, 'recall': 0, 'F1': 0, 'mIOU': 0, 'mDice': 0, 'kappa': 0}
        model.train()
        batch_num_train = 0
        batch_num_val = 0
        train_SegmentationMetric = SegmentationMetric(numClass=2)
        loss_sum_tain=0
        train_bar = tqdm(train_loader)

        #for image_A, image_B, label,edge,img_id  in train_loader:
        for image_A, image_B, label, edge, img_id in train_bar:
            batch_num_train += 1
            image_A, image_B, label ,edge= image_A.to(device), image_B.to(device), label.to(device),edge.to(device)
            model_x = model(image_A, image_B)
            # 前向传播
            label_pred=model_x[0]
            #print(model_x[1].size())
            #weight = caclulate_weight(label, 2).to(device)
            # criterion_weighted_ce.weight = weight
            # loss_ce_weighted = criterion_weighted_ce(label_pred, label)
            loss_ce=criterion_ce(label_pred, label)
            loss_iou = criterion_iou(label_pred, label, 2)
            loss_dice_edge = criterion_dice_edge(model_x[1],edge,2 )
            #loss = loss_iou +loss_ce_weighted+lamda*loss_dice_edge
            loss = loss_iou + loss_ce + lamda * loss_dice_edge
            loss_sum_tain +=loss.item()
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #更新权重
            #weight = torch.tensor([2])

            label_pred1 = torch.argmax(label_pred, dim=1)
            # print(label_pred1.size(),label.size())
            #计算精度
            train_ConfusionMatrix = train_SegmentationMetric.addBatch(label_pred1, label)
            #print(train_SegmentationMetric.confusionMatrix)



        train_OA = train_SegmentationMetric.OverallAccuary()
        train_PA = train_SegmentationMetric.PixelAccuary()
        train_IOU = train_SegmentationMetric.IntersectionOverUnion()
        train_recall = train_SegmentationMetric.recall()
        train_F1 = train_SegmentationMetric.F1()
        train_kappa = train_SegmentationMetric.kappa()
        train_loss=loss_sum_tain/batch_num_train
        print()
        print(
            "Epoch [{}/{}],train_loss: {:.4f}, OA: {:.4f}, PA: {:.4f}, IOU: {:.4f},"
            " recall_train: {:.4f}, F1_train: {:.4f}, kappa_train: {:.4f}".format(
                epoch_id,opt.num_epochs,train_loss, train_OA, train_PA, train_IOU,
                train_recall, train_F1, train_kappa))
        train_SegmentationMetric.reset()


        writer.add_scalar("train_loss_epoch", train_loss, epoch_id)
        writer.add_scalar("train_OA_epoch", train_OA, epoch_id)
        writer.add_scalar("train_pa_epoch", train_PA, epoch_id)
        writer.add_scalar("train_IOU_epoch", train_IOU, epoch_id)
        writer.add_scalar("train_recall_epoch", train_recall, epoch_id)
        writer.add_scalar("train_F1_epoch", train_F1, epoch_id)
        writer.add_scalar("train_kappa_epoch", train_kappa, epoch_id)
        result_train.append(
            [epoch_id, train_loss, train_OA, train_PA, train_IOU,
             train_recall, train_F1, train_kappa])

        ####################################val
        val_SegmentationMetric = SegmentationMetric(numClass=2)
        loss_sum_val = 0
        #val_running_results = {'loss': 0, 'PA': 0, 'mPA': 0, 'recall': 0, 'F1': 0, 'mIOU': 0, 'mDice': 0, 'kappa': 0}
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for image_A, image_B, label, edge,img_id in val_bar:
                batch_num_val += 1
                image_A, image_B, label,edge = image_A.to(device), image_B.to(device), label.to(device),edge.to(device)
                model_x = model(image_A, image_B)
                label_pred=model_x[0]
                # weight = caclulate_weight(label, 2).to(device)
                # criterion_weighted_ce.weight = weight
                loss_iou = criterion_iou(label_pred, label, 2)
                #loss_ce_weighted = criterion_weighted_ce(label_pred, label)
                loss_ce=criterion_ce(label_pred, label)
                loss_dice_edge = criterion_dice_edge(model_x[1], edge, 2)
                # loss = loss_iou + loss_ce_weighted + lamda * loss_dice_edge
                loss = loss_iou + loss_ce + lamda * loss_dice_edge
                #loss = loss_dice + loss_ce_weighted
                loss_sum_val += loss.item()
                label_pred1 = torch.argmax(label_pred, dim=1)

                # 计算精度
                val_ConfusionMatrix = val_SegmentationMetric.addBatch(label_pred1, label)




            val_OA = val_SegmentationMetric.OverallAccuary()
            val_PA = val_SegmentationMetric.PixelAccuary()
            val_IOU = val_SegmentationMetric.IntersectionOverUnion()
            val_recall = val_SegmentationMetric.recall()
            val_F1 = val_SegmentationMetric.F1()
            val_kappa = val_SegmentationMetric.kappa()
            val_loss = loss_sum_val / batch_num_val

            lr = optimizer.param_groups[0]['lr']
            lr_scheduler_model.step(val_loss)

            writer.add_scalar("lr", lr, epoch_id)
            writer.add_scalar("val_loss_epoch", val_loss, epoch_id)
            writer.add_scalar("val_OA_epoch", val_OA, epoch_id)
            writer.add_scalar("val_pa_epoch", val_PA, epoch_id)
            writer.add_scalar("val_IOU_epoch", val_IOU, epoch_id)
            writer.add_scalar("val_recall_epoch", val_recall, epoch_id)
            writer.add_scalar("val_F1_epoch", val_F1, epoch_id)
            writer.add_scalar("val_kappa_epoch", val_kappa, epoch_id)
            print()
            print(
                "Epoch [{}/{}],val_loss: {:.4f}, val_OA: {:.4f}, val_PA: {:.4f}, val_IOU: {:.4f},"
                " val_recall: {:.4f}, val_F1: {:.4f}, val_kappa: {:.4f}".format(
                    epoch_id, opt.num_epochs, val_loss, val_OA, val_PA, val_IOU,
                    val_recall, val_F1, val_kappa))
            val_SegmentationMetric.reset()

            result_val.append(
                [epoch_id, val_loss, val_OA, val_PA, val_IOU,
                    val_recall, val_F1, val_kappa])


            if not os.path.exists(save_model_dir):
                os.makedirs(save_model_dir)
            #torch.save(netCD, save_model_dir+'netCD_epoch_%d.pth' % (epoch_id))

            if val_IOU > max_IOU :
                max_IOU = val_IOU
                torch.save(model, save_model_dir+'netCD_epoch.pth' )
                print('model saved,max_IOU=',max_IOU)


            if not os.path.exists(save_csv_dir):
                os.makedirs(save_csv_dir)
            data = pd.DataFrame(data=result_train, index=None,
                                columns=['epoch', 'loss', 'oa', 'pa', 'iou',  'recall', 'F1', 'kappa'])
            # print(data)
            data.to_csv(save_csv_dir + 'train.csv')
            data = pd.DataFrame(data=result_val, index=None,
                                columns=['epoch', 'loss', 'oa', 'pa', 'iou', 'recall', 'F1', 'kappa'])
            # print(data)
            data.to_csv(save_csv_dir + 'val.csv')

    del model











