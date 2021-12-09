import glob

import numpy as np
import torch


def readFlow(fn):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
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


our_path = "/data/home/nicolashug/cluster/work/vision/sintel_submission/"
their_path = "/data/home/nicolashug/cluster/work/RAFT/sintel_submission/"

our_files = sorted(glob.glob(our_path + "**/*.flo", recursive=True))
their_files = sorted(glob.glob(their_path + "**/*.flo", recursive=True))

assert len(our_files) == len(their_files)

OK = KO = 0
for our_file, their_file in zip(our_files, their_files):
    assert our_file.replace("vision", "RAFT") == their_file
#     ours = torch.tensor(readFlow(our_file))
#     theirs = torch.tensor(readFlow(their_file))
#     try:
#         torch.testing.assert_close(ours, theirs, atol=0.5, rtol=1e-3)
#         OK += 1
#         print("OK")
#     except AssertionError as e:
#         KO += 1
#         print(our_file)
#         print("GAD", torch.max(torch.abs(ours - theirs)))

#     print()
# print(OK, OK / (OK + KO))
# print(KO, KO / (OK + KO))


# our_file = "/data/home/nicolashug/cluster/work/vision/sintel_submission/final/wall/frame0033.flo"
# their_file = our_file.replace("vision", "RAFT")
# ours = readFlow(our_file)
# theirs = readFlow(their_file)
# # print(ours[:, :, 0])
# # print()
# # print(theirs[:, :, 0])
# indices = np.abs(ours - theirs) > 10
# print(ours.shape)
# print(np.unique(np.where(indices)[-1], return_counts=True))
# print(ours[indices])
# print(theirs[indices])
