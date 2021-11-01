import os
import os.path as osp
import random
import re
from abc import ABC, abstractmethod
from glob import glob

import numpy as np
import torch
from PIL import Image

from .. import transforms as T
from ..io.image import _read_png_16
from .vision import VisionDataset


class FlowDataset(ABC, VisionDataset):
    def __init__(self, root, transforms=None):

        super().__init__(root=root)
        self.transforms = transforms

        self.init_seed = False
        self._flow_list = []
        self._image_list = []
        self._extra_info = []

    def _read_img(self, file_name):
        return Image.open(file_name)

    @abstractmethod
    def _read_flow(self, file_name):
        # Return the flow or a tuple (flow, valid) for datasets where the valid mask is built-in
        pass

    def __getitem__(self, index):
        # Some datasets like Kitti have a built-in valid mask, indicating which flow values are valid
        # For those we return (img1, img2, flow, valid), and for the rest we return (img1, img2, flow),
        # and it's up to whatever consumes the dataset to decide what `valid` should be.

        img1 = self._read_img(self._image_list[index][0])
        img2 = self._read_img(self._image_list[index][1])
        flow = self._read_flow(self._flow_list[index])

        if isinstance(flow, tuple):
            flow, valid = flow
        else:
            valid = None

        if self.transforms is not None:
            img1, img2, flow, valid = self.transforms(img1, img2, flow, valid)

        if valid is None:
            return img1, img2, flow
        else:
            return img1, img2, flow, valid

    def __len__(self):
        return len(self._image_list)


class FlyingChairs(FlowDataset):
    def __init__(
        self,
        root="/data/home/nicolashug/cluster/work/downloads/FlyingChairs_release/",
        split="training",
        transforms=None,
    ):
        super().__init__(root=root, transforms=transforms)

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

    def _read_flow(self, file_name):
        return _read_flo(file_name)


class FlyingThings3D(FlowDataset):
    def __init__(
        self,
        root="/data/home/nicolashug/cluster/work/downloads/FlyingThings3D/",
        dstype="frames_cleanpass",
        transforms=None,
    ):
        super().__init__(root=root, transforms=transforms)

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

    def _read_flow(self, file_name):
        flow = _read_pfm(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            # TODO: maybe remove this
            raise ValueError("Does this ever happen??????????????????")
            # return flow
        else:
            return flow[:, :, :-1]


class Sintel(FlowDataset):
    def __init__(
        self,
        root="/data/home/nicolashug/cluster/work/downloads/Sintel",
        split="training",  # TODO: remove??
        dstype="clean",
        transforms=None,
    ):

        super().__init__(root=root, transforms=transforms)

        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dstype)

        # if split == "test":
        #     self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self._image_list += [[image_list[i], image_list[i + 1]]]
                self._extra_info += [(scene, i)]  # scene and frame_id

            if split != "test":
                self._flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))

    def _read_flow(self, file_name):
        return _read_flo(file_name)


class KittiFlowDataset(FlowDataset):
    def __init__(
        self,
        root="/data/home/nicolashug/cluster/work/downloads/kitti",
        split="training",
        transforms=None,
    ):
        super().__init__(root=root, transforms=transforms)

        # if split == 'testing':
        #     self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, "image_2/*_10.png")))
        images2 = sorted(glob(osp.join(root, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            self._image_list += [[img1, img2]]

        if split == "training":
            self._flow_list = sorted(glob(osp.join(root, "flow_occ/*_10.png")))

    def _read_flow(self, file_name):
        return _read_16bits_png_with_flow_and_valid_mask(file_name)


def _read_flo(file_name):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(file_name, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            raise ValueError("Magic number incorrect. Invalid .flo file")

        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print 'Reading %d x %d flo file\n' % (w, h)
        data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
        # Reshape data into 3D array (columns, rows, bands)
        # The reshape here is for visualization, the original code is (w,h,2)
        return np.resize(data, (int(h), int(w), 2))


def _read_pfm(file_name):
    """Read flow in .pfm format"""
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
    return data.astype(np.float32)


def _read_16bits_png_with_flow_and_valid_mask(file_name):

    flow_and_valid = _read_png_16(file_name).to(torch.float32)
    flow, valid = flow_and_valid[:2, :, :], flow_and_valid[2, :, :]
    flow = (flow - 2 ** 15) / 64  # This conversion is explained somewhere on the kitti archive

    return flow, valid
