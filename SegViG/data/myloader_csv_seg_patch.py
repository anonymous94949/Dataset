import torch
import torch.utils.data
import torch.distributed as dist
import numpy as np

from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.random_erasing import RandomErasing
from timm.data.mixup import FastCollateMixup
from timm.data.loader import fast_collate, PrefetchLoader, MultiEpochsDataLoader

from .rasampler import RASampler
import pandas as pd
from PIL import Image
import os
import json
from torchvision.transforms import ToTensor

from scipy.ndimage import convolve1d
from utils import get_lds_kernel_window
from torch.utils.data import WeightedRandomSampler


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class PanoBufferDataset(torch.utils.data.Dataset):
    """
    Buffer-wise dataset: samples `num_images_per_buffer` images from each
    panoramic location (buffer) identified by the `no` column.

    Expected CSV columns: `no`, `q7`, `file_path`, `seg_result`
      - `no`:           integer buffer ID
      - `q7`:           walkability label (1-based)
      - `file_path`:    image path relative to data_dir
      - `seg_result`:   JSON-encoded 2D segmentation map (14×14)
    """
    def __init__(self, csv_file, data_dir=None, transform=None,
                 num_images_per_buffer=8, min_images_per_buffer=10,
                 reweight='none', lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.data = pd.read_csv(csv_file).reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.num_images_per_buffer = num_images_per_buffer
        self.min_images_per_buffer = min_images_per_buffer
        self.reweight = reweight
        self.lds = lds
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        self._prepare_weights()
        self.num_per_class = self.data['q7'].value_counts().to_dict()

        self.buffer_to_indices = {
            buffer_id: self.data.index[self.data['no'] == buffer_id].tolist()
            for buffer_id in self.data['no'].unique()
            if len(self.data.index[self.data['no'] == buffer_id]) >= self.min_images_per_buffer
        }
        self.buffer_ids = list(self.buffer_to_indices.keys())

    def _prepare_weights(self, max_target=10):
        assert self.reweight in {'none', 'inverse', 'sqrt_inv'}
        value_dict = {x: 0 for x in range(max_target)}
        labels = self.data['q7'].values - 1

        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1

        if self.reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif self.reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}

        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]

        if not len(num_per_label) or self.reweight == 'none':
            self.weights = None
            return

        if self.lds:
            lds_kernel_window = get_lds_kernel_window(self.lds_kernel, self.lds_ks, self.lds_sigma)
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]),
                weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        self.weights = [scaling * x for x in weights]

    def __len__(self):
        return len(self.buffer_ids)

    def __getitem__(self, idx):
        buffer_id = self.buffer_ids[idx]
        indices = self.buffer_to_indices[buffer_id]
        replace = len(indices) < self.num_images_per_buffer
        sampled_indices = np.random.choice(indices, size=self.num_images_per_buffer, replace=replace)

        images = []
        seg_result_list = []
        for i in sampled_indices:
            item = self.data.iloc[i]
            file_path = item['file_path']
            img_name = (os.path.join(self.data_dir, file_path)
                        if self.data_dir is not None else file_path)
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
                if not isinstance(image, torch.Tensor):
                    image = ToTensor()(image)

            seg_result = torch.tensor(json.loads(item['seg_result']), dtype=torch.float32)
            images.append(image)
            seg_result_list.append(seg_result)

        images = torch.stack(images)
        seg_result_list = torch.stack(seg_result_list)
        label = torch.tensor(self.data.loc[sampled_indices[0], 'q7'] - 1, dtype=torch.long)
        return images, seg_result_list, label


def custom_collate_fn(batch):
    """
    Flatten buffer-wise data for DataLoader.
    Input:  [(images_buf, seg_buf, label), ...]
    Output: (images [B*N, 3, H, W], (seg [B*N, H, W], labels [B]))
    """
    images_list, seg_list, labels_list = zip(*batch)
    images = torch.cat([img.view(-1, 3, 224, 224) for img in images_list], dim=0)
    seg = torch.cat(seg_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return images, (seg, labels)


class CSVLabeledDataset(torch.utils.data.Dataset):
    """
    Single-image dataset loading images and pre-computed segment maps.

    Expected CSV columns: `avg_q7`, `merged_name`
      - `avg_q7`:      walkability label (1-based float/int)
      - `merged_name`: image filename (relative to data_dir)

    Segment data is loaded from a companion directory:
      - Without overlap: <meta_dir>_segment/train_test.json
      - With overlap:    <meta_dir>_segment_overlap/train_test.json

    The JSON maps each `merged_name` to a 14×14 segment ID array.
    """
    def __init__(self, csv_file, data_dir=None, transform=None,
                 reweight='inverse', lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2,
                 overlap=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.reweight = reweight
        self.lds = lds
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        self.num_per_class = self.data['avg_q7'].value_counts().to_dict()
        self.data_dir = data_dir
        self._prepare_weights()

        meta_dir = os.path.dirname(csv_file)
        seg_suffix = "segment_overlap" if overlap else "segment"
        seg_dir = os.path.join(meta_dir, seg_suffix, "train_test.json")
        with open(seg_dir, 'r') as f:
            self.segment_data = json.load(f)

    def _prepare_weights(self, max_target=10):
        assert self.reweight in {'none', 'inverse', 'sqrt_inv'}
        value_dict = {x: 0 for x in range(max_target)}
        labels = self.data['avg_q7'].values - 1
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1

        if self.reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif self.reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}

        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        self.num_per_label = num_per_label

        if not len(num_per_label) or self.reweight == 'none':
            self.weights = None
            return

        if self.lds:
            lds_kernel_window = get_lds_kernel_window(self.lds_kernel, self.lds_ks, self.lds_sigma)
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]),
                weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        self.weights = [scaling * x for x in weights]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        file_path = item['merged_name']
        img_name = (os.path.join("../", self.data_dir, file_path)
                    if self.data_dir is not None else file_path)
        if not os.path.exists(img_name):
            img_name = img_name.replace('.png', '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = item['avg_q7'] - 1
        seg_result = torch.tensor(self.segment_data[file_path], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, (seg_result, label)


def create_loader(
        csv_file,
        input_size,
        batch_size,
        data_dir=None,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        repeated_aug=False,
        buffer_wise=False,
        num_images_per_buffer=8,
        min_images_per_buffer=10,
        vis=False,
        weighted=False,
        overlap=False,
):
    re_num_splits = 0
    if re_split:
        re_num_splits = num_aug_splits or 2

    transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    if buffer_wise:
        dataset = PanoBufferDataset(csv_file, data_dir, transform,
                                    num_images_per_buffer, min_images_per_buffer)
    else:
        dataset = CSVLabeledDataset(csv_file, data_dir, transform=transform, overlap=overlap)

    sampler = None
    if distributed:
        if is_training:
            if repeated_aug:
                sampler = RASampler(dataset, num_replicas=get_world_size(),
                                    rank=get_rank(), shuffle=True)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = OrderedDistributedSampler(dataset)
    else:
        if is_training and repeated_aug:
            sampler = RASampler(dataset, num_replicas=get_world_size(),
                                rank=get_rank(), shuffle=True)

    if buffer_wise:
        collate_fn = custom_collate_fn
    elif collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = MultiEpochsDataLoader if use_multi_epochs_loader else torch.utils.data.DataLoader

    if sampler is None and weighted:
        sampler = WeightedRandomSampler(dataset.weights, num_samples=len(dataset), replacement=True)

    loader = loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and is_training),
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
    )

    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
        )

    return loader
