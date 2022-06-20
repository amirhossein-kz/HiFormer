import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models.SwinRetina_App3 import SwinRetina_V3

from trainer import trainer_synapse
import models.SwinRetina_configs as configs 

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='./data/Synapse/test_vol_h5', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=401, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=10, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--num_workers', type=int,  default=2,
                    help='number of workers')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--output_dir', type=str,
                    default='./results', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='swin_res34_cv_11_11_612_true_cfg', help='[Swin_Res18, Swin_Res34, Swin_Res50]')
parser.add_argument('--eval_interval', type=int,
                    default=50, help='evaluation epoch')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')

args = parser.parse_args()

args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs('./logs', exist_ok=True)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }

    CONFIGS = {
        'swin_res34_cv_11_11_612_true_cfg'  : configs.get_swin_res34_cv_11_11_612_true_cfg(),
        'swin_res34_cv_120_221_612_true_cfg': configs.get_swin_res34_cv_120_221_612_true_cfg(),
        'swin_res34_cv_120_111_612_true_cfg': configs.get_swin_res34_cv_120_111_612_true_cfg(),
        'swin_res34_cv_120_221_66_true_cfg' : configs.get_swin_res34_cv_120_221_66_true_cfg(),
        'swin_res34_cv_11_11_44_true_cfg'   : configs.get_swin_res34_cv_11_11_44_true_cfg(),
        'swin_res34_cv_11_11_33_true_cfg'   : configs.get_swin_res34_cv_11_11_33_true_cfg(),
        'swin_res34_cv_110_111_612_true_cfg': configs.get_swin_res34_cv_110_111_612_true_cfg(),
        'swin_res34_cv_130_331_66_true_cfg' : configs.get_swin_res34_cv_130_331_66_true_cfg(),
        'swin_res34_cv_110_221_612_true_cfg': configs.get_swin_res34_cv_110_221_612_true_cfg(),
        'swin_res34_cv_140_441_66_true_cfg' : configs.get_swin_res34_cv_140_441_66_true_cfg(),
        'swin_res50_cv_110_111_612_true_cfg': configs.get_swin_res50_cv_110_111_612_true_cfg(),
        'swin_res50_cv_120_221_612_true_cfg': configs.get_swin_res50_cv_120_221_612_true_cfg(),
        'swin_res50_cv_120_221_66_true_cfg' : configs.get_swin_res50_cv_120_221_66_true_cfg(),
        'swin_res50_cv_130_331_66_true_cfg' : configs.get_swin_res50_cv_130_331_66_true_cfg(),
        'swin_res50_cv_120_111_612_true_cfg': configs.get_swin_res50_cv_120_111_612_true_cfg(),
        'swin_res50_cv_120_111_66_true_cfg' : configs.get_swin_res50_cv_120_111_66_true_cfg(),
        'swin_res50_cv_11_11_612_true_cfg'  : configs.get_swin_res50_cv_11_11_612_true_cfg(),
        'swin_res18_cv_11_11_612_true_cfg'  : configs.get_swin_res18_cv_11_11_612_true_cfg(),
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    net = SwinRetina_V3(config=CONFIGS[args.model_name], img_size=args.img_size, n_classes=args.num_classes).cuda()
   
    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, args.output_dir)
