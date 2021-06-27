from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

torch.backends.cudnn.benchmark = True

from dsgn.dataloader import KITTILoader3D as ls
from dsgn.dataloader import KITTILoader_dataset3d as DA
from dsgn.models import *

from env_utils import *

from dsgn.models.loss3d import RPN3DLoss

# multiprocessing distributed training
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

def get_parser():
    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('-cfg', '--cfg', '--config', default='./configs/default/config_car.py', help='config path')
    parser.add_argument('--data_path', default='./data/kitti/training/', help='data_path')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
    parser.add_argument('--loadmodel', default=None, help='load model')
    parser.add_argument('--savemodel', default=None, help='save model')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--devices', '-d', type=str, default=None)
    parser.add_argument('--lr_scale', type=int, default=40, metavar='S', help='lr scale')
    parser.add_argument('--split_file', default='./data/kitti/train.txt', help='split file')
    parser.add_argument('--btrain', '-btrain', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    ## for distributed training
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()

    if not args.devices:
        args.devices = str(np.argmin(mem_info()))

    if args.devices is not None and '-' in args.devices:
        gpus = args.devices.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.devices = ','.join(map(lambda x: str(x), list(range(*gpus))))

    if not args.dist_url:
        args.dist_url = "tcp://127.0.0.1:{}".format(random_int() % 30000)

    print('Using GPU:{}'.format(args.devices))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    return args

def main():
    args = get_parser()

    if args.debug:
        args.savemodel = './outputs/debug/'
        args.btrain = 1
        args.workers = 0

    global cfg
    exp = Experimenter(args.savemodel, cfg_path=args.cfg)
    cfg = exp.config
    
    reset_seed(args.seed)

    cfg.debug = args.debug
    cfg.warmup = getattr(cfg, 'warmup', True) if not args.debug else False

    ### distributed training ###
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: {}'.format(ngpus_per_node))
    args.ngpus_per_node = ngpus_per_node

    args.distributed = ngpus_per_node > 0 and (args.world_size > 1 or args.multiprocessing_distributed)
    args.multiprocessing_distributed = args.distributed

    if args.distributed and args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg, exp))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args, cfg, exp)

def main_process(args):
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main_worker(gpu, ngpus_per_node, args, cfg, exp):
    print("Using GPU: {} for training".format(gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    #------------------- Model -----------------------
    model = StereoNet(cfg)
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.btrain = int(args.btrain / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    elif ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    #------------------- Data Loader -----------------------
    all_left_img, all_right_img, all_left_disp, = ls.dataloader(args.data_path,
                                                                args.split_file,
                                                                depth_disp=True,
                                                                cfg=cfg,
                                                                is_train=True)

    ImageFloader = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True, split=args.split_file, cfg=cfg)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ImageFloader)
    else:
        train_sampler = None

    TrainImgLoader = torch.utils.data.DataLoader(
        ImageFloader, 
        batch_size=args.btrain, shuffle=(train_sampler is None), num_workers=args.workers, drop_last=True,
        collate_fn=BatchCollator(cfg), 
        sampler=train_sampler)

    args.max_warmup_step = min(len(TrainImgLoader), 500)

    #------------------ Logger -------------------------------------
    if main_process(args):
        logger = exp.logger
        logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        writer = exp.writer

    # ------------------------ Resume ------------------------------
    if args.loadmodel is not None:
        if main_process(args):
            logger.info('load model ' + args.loadmodel)
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'], strict=False)
        if 'optimizer' in state_dict:
            try:
                optimizer.load_state_dict(state_dict['optimizer'])
                if main_process(args):
                    logger.info('Optimizer Restored.')
            except Exception as e:
                if main_process(args):
                    logger.error(str(e))
                    logger.info('Failed to restore Optimizer')
        else:
            if main_process(args):
                logger.info('No saved optimizer.')
        if args.start_epoch is None:
            args.start_epoch = state_dict['epoch'] + 1

    if args.start_epoch is None:
        args.start_epoch = 1

    # ------------------------ Training ------------------------------
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch, args=args)

        for batch_idx, data_batch in enumerate(TrainImgLoader):
            start_time = time.time()
            if epoch == 1 and cfg.warmup and batch_idx < args.max_warmup_step:
                adjust_learning_rate(optimizer, epoch, batch_idx, args=args)

            losses = train(model, cfg, args, optimizer, **data_batch)
            loss = losses.pop('loss')

            if main_process(args):
                logger.info('%s: %s' % (args.savemodel.strip('/').split('/')[-1], args.devices))
                logger.info('Epoch %d Iter %d/%d training loss = %.3f , time = %.2f; Epoch time: %.3fs, Left time: %.3fs, lr: %.6f' % (
                    epoch, 
                    batch_idx, len(TrainImgLoader), loss, time.time() - start_time, (time.time() - start_time) * len(TrainImgLoader), 
                    (time.time() - start_time) * (len(TrainImgLoader) * (args.epochs - epoch) - batch_idx), optimizer.param_groups[0]["lr"]) )
                logger.info('losses: {}'.format(list(losses.items())))
                for lk, lv in losses.items():
                    writer.add_scalar(lk, lv, epoch * len(TrainImgLoader) + batch_idx)
                total_train_loss += loss

            if batch_idx == 100 and cfg.debug:
                break

        if main_process(args):
            logger.info('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
            savefilename = args.savemodel + '/finetune_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
                'optimizer': optimizer.state_dict()
            }, savefilename)
            logger.info('Snapshot {} epoch in {}'.format(epoch, args.savemodel))


def train(model, cfg, args, optimizer, imgL, imgR, disp_L, 
    calib=None, calib_R=None, image_indexes=None, targets=None, ious=None, labels_map=None):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    if targets is not None:
        for i in range(len(targets)):
            targets[i].bbox = targets[i].bbox.cuda()
            targets[i].box3d = targets[i].box3d.cuda()

    calibs_fu = torch.as_tensor([c.f_u for c in calib])
    calibs_baseline = torch.abs(torch.as_tensor([(c.P[0,3]-c_R.P[0,3])/c.P[0,0] for c, c_R in zip(calib, calib_R)]))
    calibs_Proj = torch.as_tensor([c.P for c in calib])
    calibs_Proj_R = torch.as_tensor([c.P for c in calib_R])

    # ---------
    mask = (disp_true > cfg.min_depth) & (disp_true <= cfg.max_depth)
    mask.detach_()
    # ---------

    losses = dict()

    outputs = model(imgL, imgR, calibs_fu, calibs_baseline, calibs_Proj, calibs_Proj_R=calibs_Proj_R)

    loss = 0.

    if getattr(cfg, 'PlaneSweepVolume', True) and cfg.loss_disp:
        depth_preds = [torch.squeeze(o, 1) for o in outputs['depth_preds']]

        disp_loss = 0.
        weight = [0.5, 0.7, 1.0]
        for i, o in enumerate(depth_preds):
            disp_loss += weight[3 - len(depth_preds) + i]  * F.smooth_l1_loss(o[mask], disp_true[mask], size_average=True)
        losses.update(disp_loss=disp_loss)
        loss += disp_loss

    if cfg.RPN3D_ENABLE:
        bbox_cls, bbox_reg, bbox_centerness = outputs['bbox_cls'], outputs['bbox_reg'], outputs['bbox_centerness']
        rpn3d_loss, rpn3d_cls_loss, rpn3d_reg_loss, rpn3d_centerness_loss = RPN3DLoss(cfg)(
            bbox_cls, bbox_reg, bbox_centerness, targets, calib, calib_R, 
            ious=ious, labels_map=labels_map)
        losses.update(rpn3d_cls_loss=rpn3d_cls_loss, 
            rpn3d_reg_loss=rpn3d_reg_loss, 
            rpn3d_centerness_loss=rpn3d_centerness_loss)
        loss += rpn3d_loss
    losses.update(loss=loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.multiprocessing_distributed:
        with torch.no_grad():
            loss_names = []
            all_losses = []
            for k in sorted(losses.keys()):
                loss_names.append(k)
                all_losses.append(losses[k])
            all_losses = torch.stack(all_losses, dim=0)
            dist.all_reduce(all_losses)
            all_losses /= args.ngpus_per_node
            reduced_losses = {k: v.item() for k, v in zip(loss_names, all_losses)}
    else:
        reduced_losses = {k: v.item() for k, v in losses.items()}

    return reduced_losses

class BatchCollator(object):
    def __init__(self, cfg):
        super(BatchCollator, self).__init__()
        self.cfg = cfg

    def __call__(self, batch):
        transpose_batch = list(zip(*batch))
        ret = dict()
        ret['imgL'] = torch.cat(transpose_batch[0], dim=0)
        ret['imgR'] = torch.cat(transpose_batch[1], dim=0)
        ret['disp_L'] = torch.stack(transpose_batch[2], dim=0)
        ret['calib'] = transpose_batch[3]
        ret['calib_R'] = transpose_batch[4]
        ret['image_indexes'] = transpose_batch[5]
        ii = 6
        if self.cfg.RPN3D_ENABLE:
            ret['targets'] = transpose_batch[ii]
            ii += 1
        if self.cfg.RPN3D_ENABLE:
            ret['ious'] = transpose_batch[ii]
            ii += 1
            ret['labels_map'] = transpose_batch[ii]
            ii += 1
        return ret

def adjust_learning_rate(optimizer, epoch, step=None, args=None):
    if epoch > 1 or step is None or step > args.max_warmup_step:
        if epoch <= args.lr_scale:
            lr = 0.001 / args.ngpus_per_node
        else:
            lr = 0.0001 / args.ngpus_per_node
    else:
        lr = 0.001 / args.ngpus_per_node
        warmup_pro = float(step) / args.max_warmup_step
        lr = lr * (warmup_pro + 1./3. * (1. - warmup_pro))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()

