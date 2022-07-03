import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from trainer import trainer_synapse
from utils import test_single_volume
from models.SwinRetina_App3 import SwinRetina_V3
from models.SwinRetina_App4 import SwinRetina_V4
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
parser.add_argument('--epoch_num', type=str,
                    default='19', help='epoch number for prediction')
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
                    default='./predictions', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='swin_res34_cv_11_11_612_true_cfg', help='[Swin_Res18, Swin_Res34, Swin_Res50]')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='evaluation epoch')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')
parser.add_argument('--is_savenii',
                    action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str,
                    default='./predictions', help='saving prediction as nii!')

args = parser.parse_args()


def inference(args, testloader, model, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


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
        'swin_res34_cv_110_111_33_true_cfg' : configs.get_swin_res34_cv_110_111_33_true_cfg(),
        'swin_res50_cv_110_111_33_true_cfg' : configs.get_swin_res50_cv_110_111_33_true_cfg(),
        'swin_res50_cv_11_11_33_true_cfg'   : configs.get_swin_res50_cv_11_11_33_true_cfg(),
        'swin_res18_cv_120_221_66_true_cfg' : configs.get_swin_res18_cv_120_221_66_true_cfg(),
        'swin_res101_cv_120_221_66_true_cfg': configs.get_swin_res101_cv_120_221_66_true_cfg(),
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    net = SwinRetina_V3(config=CONFIGS[args.model_name], img_size=args.img_size, n_classes=args.num_classes).cuda()
    
    snapshot = os.path.join('./results', args.model_name, args.model_name + '_epoch_' + args.epoch_num + '.pth')
    print(str(snapshot))
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        print("snapshot not found:")
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained HiFormer",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, args.model_name)
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    inference(args, testloader, net, test_save_path)