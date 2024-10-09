
import os

import cv2
import numpy as np


def caculate_edge(datapath,savepath):
    #print(datapath)
    train_label=datapath+'/train/label'
    test_label = datapath + '/test/label'
    val_label=datapath+'/val/label'

    # train_edge = datapath + '/train/edge'
    # test_edge = datapath + '/test/edge'
    # val_edge = datapath + '/val/edge'

    label_list=[train_label,test_label,val_label]
    #edge_list=[train_edge,test_edge,val_edge]

    #print(label_list)
    for label in label_list:
        list = os.listdir(label)
        #print(label)
        #print(list)
        savepath=label[:-5]
        #print(savepath)
        savepath=savepath+'edge'
        #print(savepath)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        # print(list)
        for i in list:
            #print(i)
            file_path = os.path.join(label, i)
            # print(file_path)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # print(image)
            if image is not None:
                sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

                sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                sobel_magnitude = np.uint8(sobel_magnitude)

                _, binary_image = cv2.threshold(sobel_magnitude, 20, 255, cv2.THRESH_BINARY)
                save_file = os.path.join(savepath, i)

                cv2.imwrite(save_file, binary_image)




if __name__=='__main__':
    caculate_edge(datapath=r'E:\change_detection_all\data\test',
                  savepath=r'E:\change_detection_all\data\test')