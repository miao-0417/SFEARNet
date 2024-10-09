import argparse
parser = argparse.ArgumentParser(description='Train Change Detection Models')


parser.add_argument('--data_dir', default=r'E:\change_detection_all\data\CLCD_256/', type=str, help='where for data.')
parser.add_argument('--result_dir', default=r'./CLCD_result/', type=str, help='where to write.')
parser.add_argument('--data', default=r'CLCD', type=str, help='which dataset')


parser.add_argument('--train_batchsize', default=8, type=int, help='batchsizefor train')
parser.add_argument('--val_batchsize', default=1, type=int, help='batchsize for validation')
parser.add_argument('--lr',  default=1e-4,type=float , help='initial learning rate for adam')
parser.add_argument('--num_epochs', default= 100, type=int, help='train epoch number')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--model', default=r'SFEARNet', type=str, help='which model')
parser.add_argument('--lr_decline', default=r'ReduceLROnPlateau', type=str, help='ReduceLROnPlateau')

parser.add_argument('--weight_decay', default=1e-3,type=float ,   help='weight_decay')

parser.add_argument('--lamda', default=3,type=float ,   help='edge_loss')


