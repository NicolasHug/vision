import argparse
import io
import pickle
from math import ceil
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default="/datasets01_ontap/tinyimagenet/081318/train/")
parser.add_argument("--output-dir", default="./tinyimagenet/081318/train")
parser.add_argument("--archive-size", type=int, default=500)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--archive-prefix", default=None)

args = parser.parse_args()

args.output_dir = Path(args.output_dir).resolve()
args.output_dir.mkdir(parents=True, exist_ok=True)


class Archiver:
    # nasty, stateful as hell, don't judge pls
    def __init__(self):
        self.dataset = torchvision.datasets.ImageFolder(args.input_dir, loader=self.loader)
        self.num_samples = len(self.dataset)

        if args.shuffle:
            self.indices = torch.randperm(self.num_samples)
        else:
            self.indices = torch.arange(self.num_samples)

    def archive_dataset(self):
        samples = []
        for i, idx in enumerate(tqdm(self.indices)):
            img, target = self.dataset[idx]
            samples.append((img, target))

            if ((i + 1) % args.archive_size == 0) or (i == len(self.indices) - 1):
                self.write_archive(samples, i)
                samples = []

    def write_archive(self, samples, i):
        current_archive_number = i // args.archive_size
        total_num_archives_needed = ceil(self.num_samples / args.archive_size)
        zero_pad_fmt = len(str(total_num_archives_needed))
        num_samples_in_archive = len(samples)

        prefix = args.archive_prefix or "archive"
        name = (
            args.output_dir / f"{prefix}_{current_archive_number:0{zero_pad_fmt}d}_{num_samples_in_archive}.pkl"
        )
        print(f"Archiving {num_samples_in_archive} samples in {name.relative_to(Path('.').resolve())}")
        with open(name, "wb") as f:
            pickle.dump(samples, f)

    def loader(self, path):
        with open(path, "rb") as f:
            # PIL can't read pure bytes, it needs a file-like object
            return io.BytesIO(f.read())

Archiver().archive_dataset()
