import numpy as np
import torch
# import warnings
# warnings.simplefilter("error")
np.seterr(divide='ignore', invalid='ignore')
#####################在这里指标计算去除了背景类的值

class SegmentationMetric(object):
    def __init__(self,numClass):
        self.numClass=numClass
        self.confusionMatrix=np.zeros((self.numClass,self.numClass))
    def OverallAccuary(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc=np.diag(self.confusionMatrix).sum()/self.confusionMatrix.sum()
        #acc[np.isnan(acc)] = 0
        return acc
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP

        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        classAcc[np.isnan(classAcc)] = 0
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
    def PixelAccuary(self):
        """
                Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
                :return:
                """

        classAcc = self.classPixelAccuracy()
        meanAcc=0
        if len(classAcc)-1!=0:
            meanAcc = np.nanmean(classAcc[1:])  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        #meanAcc[np.isnan(meanAcc)] = 0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
    def classrecall(self):
        # TP/(TP+FN)
        class_racall=np.diag(self.confusionMatrix)/self.confusionMatrix.sum(axis=1)
        class_racall[np.isnan(class_racall)] = 0
        return class_racall
    def recall(self):
        recall=0
        class_racall=self.classrecall()
        #print(class_racall)
        if len(class_racall)-1!=0:
            recall=np.nanmean(class_racall[1:])
        #recall[np.isnan(recall)] = 0
        return recall
    def classF1(self):
        classAcc = self.classPixelAccuracy()
        class_racall = self.classrecall()
        class_F1=np.divide(2*classAcc*class_racall, (classAcc+class_racall), out=np.zeros_like(classAcc),
                  where=(classAcc+class_racall)!= 0)
        #class_F1=2*classAcc*class_racall/(classAcc+class_racall+1e-7)
        class_F1[np.isnan(class_F1)] = 0
        return class_F1
    def F1(self):
        class_F1=self.classF1()
        F1=0
        if len(class_F1)-1!=0:
            F1=np.nanmean(class_F1[1:])
        #F1[np.isnan(F1)] = 0
        return F1
    def classIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        IoU[np.isnan(IoU)] = 0
        return IoU

    def IntersectionOverUnion(self):
        iou=self.classIntersectionOverUnion()
        mIoU=0
        if len(iou)-1!=0:
            mIoU = np.nanmean(iou[1:])  # 求各类别IoU的平均
        #mIoU[np.isnan(mIoU)] = 0
        return mIoU


    def kappa(self):
        """计算kappa值系数"""
        pe_rows = np.sum(self.confusionMatrix, axis=0)
        pe_cols = np.sum(self.confusionMatrix, axis=1)
        #print( pe_rows,pe_cols)
        sum_total = sum(pe_cols)

        #print(sum_total)
        pe=np.divide(np.dot(pe_rows, pe_cols), float(sum_total ** 2), out=np.zeros_like(np.dot(pe_rows, pe_cols)),
                  where=(float(sum_total ** 2))!= 0)
        #pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        #print(pe)
        po= np.divide(np.trace(self.confusionMatrix), float(sum_total), out=np.zeros_like(np.trace(self.confusionMatrix)),
                       where=(float(sum_total)) != 0)

        kappa_xishu=(po - pe) / (1 - pe+1e-7)
        #kappa_xishu[np.isnan(kappa_xishu)] = 0
        return kappa_xishu

    def getConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label.cpu(), minlength=self.numClass * self.numClass)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        #print(confusionMatrix)
        return confusionMatrix


    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.getConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵

        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

