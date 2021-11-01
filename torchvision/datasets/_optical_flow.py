import os
import os.path as osp
import random
import re
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image

from .vision import VisionDataset


class FlowDataset(VisionDataset):
    def __init__(self, root, transforms=None, sparse=False):
        # TODO: WTH does sparse mean?

        super().__init__(root=root)
        self.transforms = transforms

        self._sparse = sparse
        # self.is_test = False
        self.init_seed = False
        self._flow_list = []
        self._image_list = []
        self._extra_info = []

    def __getitem__(self, index):

        # if self.is_test:
        #     img1 = frame_utils.read_gen(self.image_list[index][0])
        #     img2 = frame_utils.read_gen(self.image_list[index][1])
        #     img1 = np.array(img1).astype(np.uint8)[..., :3]
        #     img2 = np.array(img2).astype(np.uint8)[..., :3]
        #     img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        #     img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        #     return img1, img2, self.extra_info[index]

        # TODO: Do I need to keep this??
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        # index = index % len(self._image_list)

        if self._sparse:
            # Note: See README of "development kit" archive of kitti
            flow = cv2.imread(self._flow_list[index], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            flow = flow[:, :, ::-1].astype(np.float32)
            flow, valid = flow[:, :, :2], flow[:, :, 2]
            flow = (flow - 2 ** 15) / 64.0
        else:
            flow = self._read_file(self._flow_list[index])  # TODO: change this
            valid = None

        # Note: can't use read_image, they're 16bits pngs for Kitti
        img1 = self._read_file(self._image_list[index][0])
        img2 = self._read_file(self._image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.transforms is not None:
            if self._sparse:
                img1, img2, flow, valid = self.transforms(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.transforms(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def _read_file(self, file_name):
        # TODO: probably put this out of this class

        def read_flow():
            """Read .flo file in Middlebury format"""
            # Code adapted from:
            # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

            # WARNING: this will work on little-endian architectures (eg Intel x86) only!
            # print 'fn = %s'%(fn)
            with open(file_name, "rb") as f:
                magic = np.fromfile(f, np.float32, count=1)
                if 202021.25 != magic:
                    print("Magic number incorrect. Invalid .flo file")
                    return None
                else:
                    w = np.fromfile(f, np.int32, count=1)
                    h = np.fromfile(f, np.int32, count=1)
                    # print 'Reading %d x %d flo file\n' % (w, h)
                    data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                    # Reshape data into 3D array (columns, rows, bands)
                    # The reshape here is for visualization, the original code is (w,h,2)
                    return np.resize(data, (int(h), int(w), 2))

        def read_PFM():
            file = open(file_name, "rb")

            color = None
            width = None
            height = None
            scale = None
            endian = None

            header = file.readline().rstrip()
            if header == b"PF":
                color = True
            elif header == b"Pf":
                color = False
            else:
                raise Exception("Not a PFM file.")

            dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception("Malformed PFM header.")

            scale = float(file.readline().rstrip())
            if scale < 0:  # little-endian
                endian = "<"
                scale = -scale
            else:
                endian = ">"  # big-endian

            data = np.fromfile(file, endian + "f")
            shape = (height, width, 3) if color else (height, width)

            data = np.reshape(data, shape)
            data = np.flipud(data)
            return data

        ext = osp.splitext(file_name)[-1]
        if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
            return Image.open(file_name)
        elif ext == ".flo":
            return read_flow().astype(np.float32)
        elif ext == ".pfm":
            flow = read_PFM().astype(np.float32)
            if len(flow.shape) == 2:
                raise ValueError("Does this ever happen??????????????????")
                return flow
            else:
                return flow[:, :, :-1]
        else:
            raise ValueError("Unknown file extension")

    def __rmul__(self, v):
        self._flow_list = v * self._flow_list
        self._image_list = v * self._image_list
        return self

    def __len__(self):
        return len(self._image_list)


class KittiFlowDataset(FlowDataset):
    def __init__(
        self,
        root="/data/home/nicolashug/cluster/work/downloads/kitti",  # TODO: obviously change that
        split="training",
        transforms=None,
    ):

        super().__init__(root=root, transforms=transforms, sparse=True)

        # if split == 'testing':
        #     self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, "image_2/*_10.png")))  # TODO os sep. check everywhere else
        images2 = sorted(glob(osp.join(root, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1]  # TODO os sep
            self._extra_info += [[frame_id]]
            self._image_list += [[img1, img2]]

        if split == "training":
            self._flow_list = sorted(glob(osp.join(root, "flow_occ/*_10.png")))


class FlyingChairs(FlowDataset):
    def __init__(
        self,
        root="/data/home/nicolashug/cluster/work/downloads/FlyingChairs_release/",
        split="training",
        transforms=None,
    ):
        super().__init__(root=root, transforms=transforms, sparse=False)

        images = sorted(glob(osp.join(root, "data/*.ppm")))  # TODO: os.sep
        flows = sorted(glob(osp.join(root, "data/*.flo")))
        assert len(images) // 2 == len(flows)

        # TODO: this file is not part of the original dataset, it comes from RAFT repo
        # change this. Hardcode splits when downloading the dataset?
        split_list = np.loadtxt(osp.join(root, "chairs_split.txt"), dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == "training" and xid == 1) or (split == "validation" and xid == 2):
                self._flow_list += [flows[i]]
                self._image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(
        self,
        root="/data/home/nicolashug/cluster/work/downloads/FlyingThings3D/",
        dstype="frames_cleanpass",
        transforms=None,
    ):
        super().__init__(root=root, transforms=transforms, sparse=False)

        cam = "left"  # TODO: Use both cams?
        for direction in ["into_future", "into_past"]:
            image_dirs = sorted(glob(osp.join(root, dstype, "TRAIN/*/*")))
            image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

            flow_dirs = sorted(glob(osp.join(root, "optical_flow/TRAIN/*/*")))
            flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

            for idir, fdir in zip(image_dirs, flow_dirs):
                images = sorted(glob(osp.join(idir, "*.png")))
                flows = sorted(glob(osp.join(fdir, "*.pfm")))
                for i in range(len(flows) - 1):
                    if direction == "into_future":
                        self._image_list += [[images[i], images[i + 1]]]
                        self._flow_list += [flows[i]]
                    elif direction == "into_past":
                        self._image_list += [[images[i + 1], images[i]]]
                        self._flow_list += [flows[i + 1]]


class Sintel(FlowDataset):
    def __init__(
        self,
        root="/data/home/nicolashug/cluster/work/downloads/Sintel",
        split="training",  # TODO: remove??
        dstype="clean",
        transforms=None,
    ):

        super().__init__(root=root, transforms=transforms, sparse=False)

        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)

        if split == "test":
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self._image_list += [[image_list[i], image_list[i + 1]]]
                self._extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self._flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))
