import argparse
from glob import glob
import os
import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import numpy as np
from PIL import Image
import cv2

from torchvision.models.video import RAFT
from torchvision.models.video._raft.utils import KittiFlowDataset


MAX_FLOW = 400


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'flow_loss': flow_loss,
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


SUM_FREQ = 50
class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}  # TODO: could be a defaultdict(int)
        self.writer = None

    def _print_training_status(self):
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ""
        for metric, val in self.running_loss.items():
            avg_val = val / SUM_FREQ
            metrics_str += f"{metric}: {avg_val:10.4f}, "
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    print(f"{torch.cuda.device_count()} cuda devices available")
    cuda_available = torch.cuda.is_available()

    model = RAFT()  # TODO: pass args

    # TODO: use DistributedDataParallel instead
    # if cuda_available:
    #     model = torch.nn.DataParallel(model)
    # print("Parameter Count: %d" % count_parameters(model))

    # if args.restore_ckpt is not None:
    #     model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    if cuda_available:
        model.cuda()
    model.train()

    # if args.stage != 'chairs':
    #     model.module.freeze_bn()

    # train_loader = datasets.fetch_dataloader(args)
    # optimizer, scheduler = fetch_optimizer(args, model)

    train_dataset = KittiFlowDataset(root="/data/home/nicolashug/cluster/work/kitti/s3.eu-central-1.amazonaws.com/avg-kitti")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')


    total_steps = 0
    # TODO: only use GradSclaer if cuda is available
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            # print(f"{i_batch = }")
            optimizer.zero_grad()
            if cuda_available:
                image1, image2, flow, valid = [x.cuda() for x in data_blob]
            else:
                image1, image2, flow, valid = data_blob

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            # print("before forward")
            flow_predictions = model(image1, image2, iters=args.iters)
            # print("done")

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                # results = {}
                # for val_dataset in args.validation:
                #     if val_dataset == 'chairs':
                #         results.update(evaluate.validate_chairs(model.module))
                #     elif val_dataset == 'sintel':
                #         results.update(evaluate.validate_sintel(model.module))
                #     elif val_dataset == 'kitti':
                #         results.update(evaluate.validate_kitti(model.module))

                # logger.write_dict(results)
                # model.train()
                # if args.stage != 'chairs':
                #     model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                print("BREAKING OUT OF TRAINING LOOP")
                should_keep_training = False
                break

    # logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not osp.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
