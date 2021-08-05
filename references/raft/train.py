import argparse
from glob import glob
import os
import os.path as osp

import torch
from torch.cuda.amp import GradScaler
import numpy as np
from PIL import Image

from torchvision.models.video import RAFT
from torchvision.models.video._raft.utils import InputPadder


class KittiFlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root='/Users/nicolashug/Downloads/data_scene_flow/',  # TODO: obviously change that
        split='training',
    ):

        # self.is_test = False
        # self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        # if split == 'testing':
        #     self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))  # TODO os sep
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]  # TODO os sep
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

    def __getitem__(self, index):

        # if self.is_test:
        #     img1 = frame_utils.read_gen(self.image_list[index][0])
        #     img2 = frame_utils.read_gen(self.image_list[index][1])
        #     img1 = np.array(img1).astype(np.uint8)[..., :3]
        #     img2 = np.array(img2).astype(np.uint8)[..., :3]
        #     img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        #     img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        #     return img1, img2, self.extra_info[index]

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.image_list)
        # valid = None
        # if self.sparse:
        #     flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        # else:

        # Can't use read_image, they're 16bits pngs
        flow = Image.open(self.flow_list[index])

        img1 = Image.open(self.image_list[index][0])
        img2 = Image.open(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # if self.augmentor is not None:
        #     if self.sparse:
        #         img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
        #     else:
        #         img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        # if valid is not None:
        #     valid = torch.from_numpy(valid)
        # else:
        valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        # padder = InputPadder(img1.shape, mode='kitti')
        # # img1, img2, flow = padder.pad(img1[None], img2[None], flow[None])
        # img1, img2, flow = padder.pad(img1, img2, flow)

        return img1, img2, flow, valid.float()

    # def __rmul__(self, v):
    #     self.flow_list = v * self.flow_list
    #     self.image_list = v * self.image_list
    #     return self

    def __len__(self):
        return len(self.image_list)


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
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def train(args):

    # model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    # print("Parameter Count: %d" % count_parameters(model))
    model = RAFT()

    # if args.restore_ckpt is not None:
    #     model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    # model.cuda()
    model.train()

    # if args.stage != 'chairs':
    #     model.module.freeze_bn()

    # train_loader = datasets.fetch_dataloader(args)
    # optimizer, scheduler = fetch_optimizer(args, model)

    train_dataset = KittiFlowDataset()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')


    total_steps = 0
    # TODO: only use GradSclaer if cuda is available
    scaler = GradScaler(enabled=args.mixed_precision)
    # logger = Logger(model, scheduler)

    # VAL_FREQ = 5000

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            print(i_batch)
            optimizer.zero_grad()
            # image1, image2, flow, valid = [x.cuda() for x in data_blob]
            image1, image2, flow, valid = data_blob

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            print("before forward")
            flow_predictions = model(image1, image2, iters=args.iters)
            print("done")

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # logger.push(metrics)

            # if total_steps % VAL_FREQ == VAL_FREQ - 1:
            #     PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
            #     torch.save(model.state_dict(), PATH)

            #     results = {}
            #     for val_dataset in args.validation:
            #         if val_dataset == 'chairs':
            #             results.update(evaluate.validate_chairs(model.module))
            #         elif val_dataset == 'sintel':
            #             results.update(evaluate.validate_sintel(model.module))
            #         elif val_dataset == 'kitti':
            #             results.update(evaluate.validate_kitti(model.module))

            #     logger.write_dict(results)

            #     model.train()
            #     if args.stage != 'chairs':
            #         model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
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
