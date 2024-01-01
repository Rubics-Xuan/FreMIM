# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train.py

import argparse
import os
import random
import logging
import numpy as np
import time
import math
import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
from tools import criterions

from data_augmentation.BraTS_pre_training import BraTS_fore
from torch.utils.data import DataLoader

from torch import nn
from models.Swin_UNTER.pretrain_frequency_2d import SwinUNETR

import warnings
warnings.filterwarnings('ignore')
import time
from tools.loss import FocalFrequencyLoss_2D_pooling

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_args_parser():
    parser = argparse.ArgumentParser('Med-MFM', add_help=False)

    parser.add_argument('--user', default='wwx', type=str)

    parser.add_argument('--dataset', default='BraTS2019', type=str)

    parser.add_argument('--mask_ratio', default=0.25, type=float)

    parser.add_argument('--model', default='UNet_16C_id5042', type=str)

    parser.add_argument('--stage_name', default='pretrain', type=str)

    # DataSet Information
    parser.add_argument('--root', default='/data/wangwenxuan/Datasets/BraTS2019/', type=str)

    parser.add_argument('--train_dir', default='Train', type=str)

    parser.add_argument('--valid_dir', default='Train', type=str)

    parser.add_argument('--passband', default=10, type=int)

    parser.add_argument('--frequncy_channels', default=16, type=int)

    parser.add_argument('--output_dir', default='output', type=str)

    parser.add_argument('--submission', default='submission', type=str)

    parser.add_argument('--visual', default='visualization', type=str)

    parser.add_argument('--heatmap_dir', default='heatmap', type=str)

    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)


    parser.add_argument('--mode', default='train', type=str)

    parser.add_argument('--train_file', default='train_0.txt', type=str)

    parser.add_argument('--valid_file', default='valid_0.txt', type=str)


    parser.add_argument('--input_C', default=4, type=int)

    parser.add_argument('--input_H', default=240, type=int)

    parser.add_argument('--input_W', default=240, type=int)

    parser.add_argument('--input_D', default=160, type=int)

    parser.add_argument('--crop_size', default=128, type=int)

    parser.add_argument('--crop_D', default=128, type=int)

    parser.add_argument('--output_D', default=155, type=int)

    # Training Information
    parser.add_argument('--lr', default=0.0001, type=float)

    parser.add_argument('--weight_decay', default=1e-5, type=float)

    parser.add_argument('--amsgrad', default=True, type=bool)

    parser.add_argument('--criterion', default='softmax_dice', type=str)

    parser.add_argument('--num_class', default=4, type=int)

    parser.add_argument('--seed', default=1024, type=int)

    parser.add_argument('--no_cuda', default=False, type=bool)

    parser.add_argument('--gpu', default='0,1,2,3', type=str)

    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--start_epoch', default=0, type=int)

    parser.add_argument('--end_epoch', default=250, type=int)

    parser.add_argument('--save_freq', default=250, type=int)

    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--load', default=True, type=bool)

    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    return parser

def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    # print('All_voxels:240*240*155 | numerator:{} | denominator:{} | pred_voxels:{} | GT_voxels:{}'.format(int(num),int(den),o.sum(),int(t.sum())))
    return num/den

def softmax_output_dice(output, target):
    ret = []

    # WT
    o = output > 0
    t = target > 0  # ce
    ret += dice_score(o, t),
    # TC
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 4)
    ret += dice_score(o, t),
    # ET
    o = (output == 3)
    t = (target == 4)
    ret += dice_score(o, t),

    return ret

def main_worker(args):
    experiment_name = f'{args.dataset}_bs{args.batch_size}_epoch{args.end_epoch}_{args.model}_{args.mask_ratio}_{args.stage_name}_{args.date}'
    if is_main_process():  # Tensorboard configuration
        output_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'train_output', args.stage_name, experiment_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_dir = os.path.join(output_dir, 'log')
        time_tuple = time.localtime(time.time())
        log_file = log_dir + f'{time_tuple[0]}_{time_tuple[1]}_{time_tuple[2]}_{time_tuple[3]}_{time_tuple[4]}.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(experiment_name))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    torch.distributed.init_process_group('nccl')      
    torch.cuda.set_device(args.local_rank)             


    pass_manners = ['high', 'low']
    loss_weight = [1, 1.3]
    mask_ratio_list = [args.mask_ratio, args.mask_ratio, args.mask_ratio]
    model = SwinUNETR(img_size=(args.crop_size, args.crop_size),
                in_channels=4,
                out_channels=4,
                feature_size=48,
                frequency_channels=args.frequncy_channels,
                whether_mask=True,
                pixel_mask_rate=args.mask_ratio)
    # ------------------------------------resume and contiune trainging---------------------------------------#
    resume = ''
    if args.local_rank == 0:
        if os.path.isfile(resume) and args.load:
            logging.info('loading checkpoint {}'.format(resume))
            checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                        .format(args.resume, args.start_epoch))
        else:
            logging.info('re-training!!!')

    # ------------------------------------DDP train---------------------------------------#
    model.cuda(args.local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)


    criterion = FocalFrequencyLoss_2D_pooling(log_matrix=True, pass_band=args.passband)

    ave_spectrum_list = [False, False]
    stage_manners = ['low_level','high_level']

    if args.local_rank == 0:
        checkpoint_dir = output_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    # ------------------------------------dataset and dataloader---------------------------------------#
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)

    train_set = BraTS_fore(train_list, train_root, args.mode)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    logging.info('Samples for train = {}'.format(len(train_set)))
    num_gpu = (len(args.gpu)+1) // 2
    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    # ------------------------------------other---------------------------------------#
    start_time_training = time.time()
    num_iter_perepoch = len(train_set) // args.batch_size * args.crop_D
    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch):
        train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(f'{args.user}: {args.stage_name}', epoch+1, args.end_epoch))
        start_epoch = time.time()
        if epoch >= 0 and epoch < int(args.end_epoch / 3):
            mask_ratio = mask_ratio_list[0]
        elif epoch >= int(args.end_epoch / 3) and epoch < 2 * int(args.end_epoch / 3):
            mask_ratio = mask_ratio_list[1]
        else:
            mask_ratio = mask_ratio_list[2]

        for i, data in enumerate(train_loader):


            _, x, target = data
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            

            max_number = (args.batch_size//num_gpu) * args.crop_D
            index = torch.randperm(max_number)
            B, D = x.size(0), x.size(-1)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(B*D, 4, args.crop_size, args.crop_size)
            target = target.permute(0, 3, 1, 2).contiguous().view(B*D, args.crop_size, args.crop_size)
            x = x[index]
            target = target[index]

            for s in range(args.crop_D):
                current_iter = epoch * num_iter_perepoch + i * int(args.crop_D) + (s + 1)
                warm_up_learning_rate_adjust_iter(args.lr, current_iter, args.end_epoch // 100 * num_iter_perepoch, args.end_epoch * num_iter_perepoch, optimizer, power=0.9)

                x_s = x[s*(args.batch_size//num_gpu):(s+1)*(args.batch_size//num_gpu), ...]

                decoder_prediction = model(x_s, mask_ratio)

                loss, loss_dict = compute_loss(decoder_prediction, x_s, criterion, pass_manners, stage_manners, loss_weight, ave_spectrum_list)

                reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if args.local_rank == 0:
                logging.info('Epoch: {}_Iter: {}_ratio: {} loss: {:.5f} || high: {:.4f} {}: {:.4f}||'
                            .format(epoch, i, mask_ratio, reduce_loss, loss_dict['high_0'].item(), pass_manners[1], loss_dict[f'{pass_manners[1]}_1'].item()))


        end_epoch = time.time()

        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

    if args.local_rank == 0:

        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.module.state_dict(),                   
            'optim_dict': optimizer.state_dict(),
        },
            final_name)

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))


    end_time = time.time()
    training_total_time = (end_time - start_time_training) / 3600
    logging.info('The total pre-training time is {:.2f} hours'.format(training_total_time))
    logging.info('----------------------------------The training process finished!-----------------------------------')


def warm_up_learning_rate_adjust_iter(init_lr, cur_iter, warmup_iter, max_iter, optimizer, power=0.9):
    for param_group in optimizer.param_groups:
        if cur_iter < warmup_iter:
            param_group['lr'] = init_lr * cur_iter / (warmup_iter+1e-8)
        else:
            param_group['lr'] = init_lr * ((1 - float(cur_iter - warmup_iter) / (max_iter - warmup_iter))**(power))



def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor


def compute_loss(decoder_prediction, image, criterion, pass_manners, stage_manners, loss_weight, ave_spectrum_list):
    total_loss = 0
    loss_dict = {}
    scale_H, scale_W = image.shape[2:]
    for i, prediction in enumerate(decoder_prediction):
        loss = criterion(prediction, image, pass_manner=pass_manners[i], stage_manner=stage_manners[i], ave_spectrum=ave_spectrum_list[i])
        loss = loss * loss_weight[i]
        loss_dict[f'{pass_manners[i]}_{i}'] = loss
        total_loss += loss
    return total_loss, loss_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Med-MFM', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    main_worker(args)
