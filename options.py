import argparse
from utils import str2bool

parser = argparse.ArgumentParser()
# training settings
parser.add_argument('--epoch', type=int, default=150, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=704, help='training dataset size')
parser.add_argument('--size2', type=int, default=120, help='training dataset size')  # input image size 352
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')

# device settings
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# dataset settings
parser.add_argument('--img_root', type=str, default='/root/autodl-fs/TrainingSets/Image/',
                    help='the training images root')
parser.add_argument('--edge_root', type=str, default='/root/autodl-fs/TrainingSets/edge/',
                    help='the training edge root')
parser.add_argument('--gt_root', type=str, default='/root/autodl-fs/TrainingSets/GT/',
                    help='the training gt images root')
parser.add_argument('--test_img_root', type=str,
                    default='/root/autodl-fs/TestingSets/CAMO/Image/',
                    help='the validation images root')
parser.add_argument('--test_gt_root', type=str,
                    default='/root/autodl-fs/TestingSets/CAMO/GT/',
                    help='the validation gt images root')
parser.add_argument('--test_path', type=str, default=r'E:\COD_datasets\TestingSets')
# parser.add_argument('--test_path', type=str, default='./TestingSets')
parser.add_argument('--test_save_path', type=str, default='./results')
parser.add_argument('--test_edge_path', type=str, default='./edge_results')
parser.add_argument('--test_name', type=str, default='run-1')

# save settings
parser.add_argument('--save_path', type=str, default='./Paper13_work_dir/',
                    help='the path to save models and logs')

# architecture settings
parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'res2net'])
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--ratio', type=float, default=0.25)
parser.add_argument('--expand_ratio', type=float, default=1.2)
parser.add_argument('--n', type=float, default=4)
parser.add_argument('--groups', type=int, default=0)
parser.add_argument('--norm', type=str, default='gn', choices=['bn', 'gn'])
parser.add_argument('--fusion', type=str, default='att', choices=['cat', 'bfm', 'att', 'spt', 'woedge', 'base'])
parser.add_argument('--fusion2', type=str, default='add', choices=['3d', 'cat', 'add', 'base'])
parser.add_argument('--enhance', type=str, default='MFEM', choices=['MFEM', 'RFB', 'fam'])
parser.add_argument('--adjust', type=str, default='stage', choices=['stage', 'poly'])
parser.add_argument('--sakernel', type=int, default=31, choices=[3, 7, 15, 31, 63])
parser.add_argument('--kernel', type=int, default=3, choices=[3, 5, 7])
parser.add_argument('--dilate', type=int, default=1, choices=[1, 2, 3])
parser.add_argument('--satype', type=str, default='sa', choices=['sa', 'sa2'])
parser.add_argument('--edge_branch', type=int, default=1, choices=[1, 2])

# loss settings
parser.add_argument('--mask_loss', type=str, default='f3', choices=['bi', 'bas', 'f3'])
parser.add_argument('--edge_loss', type=str, default='f3', choices=['bi', 'bas', 'f3', 'dice'])

opt = parser.parse_args()
