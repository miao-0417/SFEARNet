import random
import cv2
#random.seed(42)
class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1, img2, label,edge = sample
        if random.random() < 0.5:
            #print("RandomHorizontalFlip")
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
            label = cv2.flip(label, 1)
            edge=cv2.flip(edge,1)
        return img1, img2, label,edge

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1, img2, label,edge = sample
        if random.random() < 0.5:
            #print("RandomVerticalFlip")
            img1 = cv2.flip(img1, 0)
            img2 = cv2.flip(img2, 0)
            label = cv2.flip(label, 0)
            edge=cv2.flip(edge,0)
        return img1, img2, label,edge

class RandomFixRotate(object):
    def __init__(self):
        self.degree=random.randint(0,360)
        #self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1, img2, label ,edge= sample
        if random.random() < 0.3:
            #print(self.degree)
            rows, cols = img1.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.degree, 1)
            img1 = cv2.warpAffine(img1, rotation_matrix, (cols, rows))
            img2 = cv2.warpAffine(img2, rotation_matrix, (cols, rows))
            label = cv2.warpAffine(label, rotation_matrix, (cols, rows))
            edge = cv2.warpAffine(edge, rotation_matrix, (cols, rows))
            # rotate_degree = random.choice(self.degree)
            # img1 = img1.transpose(rotate_degree)
            # img2 = img2.transpose(rotate_degree)
            # label = label.transpose(rotate_degree)
        return img1, img2, label,edge

class RandomExchangeOrder(object):
    def __call__(self, sample):
        img1, img2, label,edge = sample
        if random.random() < 0.3:
            #print("RandomExchangeOrder")
            return img1, img2, label,edge
        return img1, img2, label,edge



