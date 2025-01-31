# -*- coding: utf-8 -*-
"""
Module for Semantic Segmentation

This module implements the components required for semantic segmentation:
- A custom Dataset class for PurdueShapes5MultiObject data
- Skip-block modules for downsampling and upsampling
- A U-Net–like architecture (mUnet)
- Several loss functions (MSE, Dice, and a weighted combination)
- Training and testing routines

Originally created in Colab, this version has been refactored as a standalone module.
"""

import os
import sys
import time
import copy
import random
import gzip
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import requests
from pycocotools.coco import COCO

# We assume that DLStudio (version 2.3.6) is installed and available as a package.
from DLStudio import *


# =============================================================================
# Dataset class
# =============================================================================
class PurdueShapes5MultiObjectDataset(torch.utils.data.Dataset):
    def __init__(self, dl_studio, segmenter, mode, dataset_filename):
        """
        Args:
            dl_studio: An instance of DLStudio (providing parameters like dataroot, image_size, etc.)
            segmenter: An instance (or container) that has max_num_objects defined.
            mode: 'train' or 'test'
            dataset_filename: Name of the gzipped dataset file.
        """
        super(PurdueShapes5MultiObjectDataset, self).__init__()
        self.image_size = dl_studio.image_size
        self.max_num_objects = segmenter.max_num_objects

        # For the training dataset, try to load saved torch files if they exist.
        if mode == 'train' and dataset_filename == "PurdueShapes5MultiObject-10000-train.gz":
            dataset_pt = "torch_saved_PurdueShapes5MultiObject-10000_dataset.pt"
            label_map_pt = "torch_saved_PurdueShapes5MultiObject_label_map.pt"
            if os.path.exists(dataset_pt) and os.path.exists(label_map_pt):
                print("\nLoading training data from saved torch files...")
                self.dataset = torch.load(dataset_pt)
                self.label_map = torch.load(label_map_pt)
                self.num_shapes = len(self.label_map)
            else:
                print("Loading training dataset for the first time. This may take a few minutes...")
                root_dir = dl_studio.dataroot
                with gzip.open(os.path.join(root_dir, dataset_filename), 'rb') as f:
                    raw_data = f.read()
                self.dataset, self.label_map = pickle.loads(raw_data, encoding='latin1')
                torch.save(self.dataset, dataset_pt)
                torch.save(self.label_map, label_map_pt)
                self.num_shapes = len(self.label_map)
        else:
            root_dir = dl_studio.dataroot
            with gzip.open(os.path.join(root_dir, dataset_filename), 'rb') as f:
                raw_data = f.read()
            if sys.version_info[0] == 3:
                self.dataset, self.label_map = pickle.loads(raw_data, encoding='latin1')
            else:
                self.dataset, self.label_map = pickle.loads(raw_data)
            self.num_shapes = len(self.label_map)

        # Reverse the key-value pairs: now mapping label id to class name.
        self.class_labels = {v: k for k, v in self.label_map.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        H, W = self.image_size
        # Each dataset entry is assumed to have the following structure:
        # [r_channel, g_channel, b_channel, mask, bbox_map]
        entry = self.dataset[index]

        # Convert the R, G, B channels into a 3 x H x W tensor.
        r = np.array(entry[0]).reshape(H, W)
        g = np.array(entry[1]).reshape(H, W)
        b = np.array(entry[2]).reshape(H, W)
        image_tensor = torch.stack([torch.from_numpy(r),
                                    torch.from_numpy(g),
                                    torch.from_numpy(b)], dim=0).float()

        # Create the mask tensor.
        mask_array = np.array(entry[3])
        mask_tensor = torch.from_numpy(mask_array).float()

        # Create the bounding-box tensor.
        # The original code uses the length of mask_array[0] as max number of objects.
        max_bboxes = len(mask_array[0])
        bbox_tensor = torch.zeros(self.max_num_objects, self.num_shapes, 4, dtype=torch.float)
        bbox_map = entry[4]
        for label, bbox_list in bbox_map.items():
            for idx_bbox, bbox in enumerate(bbox_list):
                if idx_bbox < max_bboxes:
                    bbox_tensor[idx_bbox, label, :] = torch.tensor(bbox, dtype=torch.float)

        return {'image': image_tensor,
                'mask_tensor': mask_tensor,
                'bbox_tensor': bbox_tensor}


# =============================================================================
# Skip-connection blocks for Downsampling and Upsampling
# =============================================================================
class SkipBlockDN(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, use_skip=True):
        """
        Downsampling skip block.
        """
        super(SkipBlockDN, self).__init__()
        self.downsample = downsample
        self.use_skip = use_skip
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if self.downsample:
            self.downsampler = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        if self.in_channels == self.out_channels:
            out = self.conv2(out)
            out = self.bn2(out)
            out = nn.ReLU()(out)
        if self.downsample:
            out = self.downsampler(out)
            identity = self.downsampler(identity)
        if self.use_skip:
            if self.in_channels == self.out_channels:
                out = out + identity
            else:
                out = out + torch.cat((identity, identity), dim=1)
        return out


class SkipBlockUP(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, use_skip=True):
        """
        Upsampling skip block.
        """
        super(SkipBlockUP, self).__init__()
        self.upsample = upsample
        self.use_skip = use_skip
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convT1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(out_channels)
        self.convT2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(out_channels)
        if self.upsample:
            self.upsampler = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1,
                                                stride=2, dilation=2,
                                                output_padding=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.convT1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        if self.in_channels == self.out_channels:
            out = self.convT2(out)
            out = self.bn2(out)
            out = nn.ReLU()(out)
        if self.upsample:
            out = self.upsampler(out)
            identity = self.upsampler(identity)
        if self.use_skip:
            if self.in_channels == self.out_channels:
                out = out + identity
            else:
                out = out + identity[:, self.out_channels:, :, :]
        return out


# =============================================================================
# mUnet Architecture (a U-Net–like network)
# =============================================================================
class mUnet(nn.Module):
    def __init__(self, skip_connections=True, depth=16):
        """
        A U-Net–like architecture for semantic segmentation.
        Args:
            skip_connections (bool): Whether to use skip connections.
            depth (int): Determines the number of skip blocks.
        """
        super(mUnet, self).__init__()
        self.half_depth = depth // 2
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Downsampling path (64-channel blocks)
        self.skip64_dn_blocks = nn.ModuleList([
            SkipBlockDN(64, 64, use_skip=skip_connections) for _ in range(self.half_depth)
        ])
        self.downsample64 = SkipBlockDN(64, 64, downsample=True, use_skip=skip_connections)
        self.transition64_to_128 = SkipBlockDN(64, 128, use_skip=skip_connections)

        # Downsampling path (128-channel blocks)
        self.skip128_dn_blocks = nn.ModuleList([
            SkipBlockDN(128, 128, use_skip=skip_connections) for _ in range(self.half_depth)
        ])
        self.downsample128 = SkipBlockDN(128, 128, downsample=True, use_skip=skip_connections)
        self.bn_down_1 = nn.BatchNorm2d(64)
        self.bn_down_2 = nn.BatchNorm2d(128)

        # Upsampling path
        self.upsample128 = SkipBlockUP(128, 128, upsample=True, use_skip=skip_connections)
        self.skip128_up_blocks = nn.ModuleList([
            SkipBlockUP(128, 128, use_skip=skip_connections) for _ in range(self.half_depth)
        ])
        self.transition128_to_64 = SkipBlockUP(128, 64, use_skip=skip_connections)
        self.skip64_up_blocks = nn.ModuleList([
            SkipBlockUP(64, 64, use_skip=skip_connections) for _ in range(self.half_depth)
        ])
        self.upsample64 = SkipBlockUP(64, 64, upsample=True, use_skip=skip_connections)
        self.bn_up_1 = nn.BatchNorm2d(128)
        self.bn_up_2 = nn.BatchNorm2d(64)
        self.conv_out = nn.ConvTranspose2d(64, 5, kernel_size=3, stride=2,
                                             dilation=2, output_padding=1, padding=2)

    def forward(self, x):
        # Initial convolution and pooling.
        x = self.conv_in(x)
        x = nn.ReLU()(x)
        x = self.pool(x)

        # Downsampling stage with 64-channel blocks.
        quarter_depth = self.half_depth // 4
        # Process first part of the 64-channel blocks.
        for block in self.skip64_dn_blocks[:quarter_depth]:
            x = block(x)
        saved1 = x[:, :x.shape[1] // 2, :, :].clone()  # Save features for later.
        x = self.downsample64(x)
        for block in self.skip64_dn_blocks[quarter_depth:]:
            x = block(x)
        x = self.bn_down_1(x)
        saved2 = x[:, :x.shape[1] // 2, :, :].clone()
        x = self.transition64_to_128(x)

        # Downsampling stage with 128-channel blocks.
        for block in self.skip128_dn_blocks[:quarter_depth]:
            x = block(x)
        x = self.bn_down_2(x)
        saved3 = x[:, :x.shape[1] // 2, :, :].clone()
        for block in self.skip128_dn_blocks[quarter_depth:]:
            x = block(x)
        x = self.downsample128(x)

        # Upsampling path.
        x = self.upsample128(x)
        for block in self.skip128_up_blocks[:quarter_depth]:
            x = block(x)
        x[:, :x.shape[1] // 2, :, :] = saved3  # Restore saved features.
        x = self.bn_up_1(x)
        for block in self.skip128_up_blocks[:quarter_depth]:
            x = block(x)
        x = self.transition128_to_64(x)
        for block in self.skip64_up_blocks[quarter_depth:]:
            x = block(x)
        x[:, :x.shape[1] // 2, :, :] = saved2
        x = self.bn_up_2(x)
        x = self.upsample64(x)
        for block in self.skip64_up_blocks[:quarter_depth]:
            x = block(x)
        x[:, :x.shape[1] // 2, :, :] = saved1
        x = self.conv_out(x)
        return x


# =============================================================================
# Loss functions
# =============================================================================
class SegmentationLossMSE(nn.Module):
    def __init__(self, batch_size):
        super(SegmentationLossMSE, self).__init__()
        self.batch_size = batch_size

    def forward(self, output, mask_tensor):
        # Compute the mean squared error between output and ground-truth masks.
        return torch.mean((output - mask_tensor) ** 2)


class SegmentationLossDice(nn.Module):
    def __init__(self, batch_size):
        super(SegmentationLossDice, self).__init__()
        self.batch_size = batch_size

    def forward(self, output, mask_tensor):
        # Compute Dice loss: 1 - (2 * (output * mask) + 1) / (output + mask + 1)
        dice = 1 - (2 * output * mask_tensor + 1) / (output + mask_tensor + 1)
        return torch.mean(dice)


class SegmentationLossComb(nn.Module):
    def __init__(self, batch_size, dice_scale):
        super(SegmentationLossComb, self).__init__()
        self.batch_size = batch_size
        self.dice_scale = dice_scale

    def forward(self, output, mask_tensor):
        # Combined loss: scaled Dice loss plus MSE loss.
        dice_loss = SegmentationLossDice(self.batch_size)(output, mask_tensor)
        mse_loss  = SegmentationLossMSE(self.batch_size)(output, mask_tensor)
        return self.dice_scale * dice_loss + mse_loss


# =============================================================================
# Main SemanticSegmentation class (Training and Testing routines)
# =============================================================================
class SemanticSegmentation:
    # Expose mUnet as an attribute so that it can be accessed via self.segmenter.mUnet(...)
    mUnet = mUnet

    def __init__(self, dl_studio, max_num_objects,
                 dataserver_train=None, dataserver_test=None,
                 dataset_file_train=None, dataset_file_test=None):
        """
        Args:
            dl_studio: An instance providing configuration (device, batch_size, etc.).
            max_num_objects: Maximum number of objects per image.
            dataserver_train: Training dataset instance.
            dataserver_test: Testing dataset instance.
            dataset_file_train: Name of the training dataset file.
            dataset_file_test: Name of the testing dataset file.
        """
        self.dl_studio = dl_studio
        self.max_num_objects = max_num_objects
        self.dataserver_train = dataserver_train
        self.dataserver_test = dataserver_test

    def setup_dataloaders(self, train_dataset, test_dataset):
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.dl_studio.batch_size,
            shuffle=True,
            num_workers=4
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.dl_studio.batch_size,
            shuffle=False,
            num_workers=4
        )

    def train_segmentation(self, net, loss_type, dice_scale=0):
        """
        Train the segmentation network.
        Args:
            net: The segmentation network (e.g. an instance of mUnet).
            loss_type: One of "MSE", "DICE", or "COMBINED" (case insensitive).
            dice_scale: Scaling factor for the dice loss if loss_type is "COMBINED".
        """
        performance_file = f"performance_numbers_{self.dl_studio.epochs}.txt"
        with open(performance_file, 'w') as fout:
            model = copy.deepcopy(net).to(self.dl_studio.device)
            loss_type = loss_type.upper()
            if loss_type == "MSE":
                criterion = SegmentationLossMSE(self.dl_studio.batch_size)
            elif loss_type == "DICE":
                criterion = SegmentationLossDice(self.dl_studio.batch_size)
            elif loss_type == "COMBINED":
                criterion = SegmentationLossComb(self.dl_studio.batch_size, dice_scale)
            else:
                raise ValueError("Invalid loss type specified.")

            optimizer = optim.SGD(model.parameters(),
                                  lr=self.dl_studio.learning_rate,
                                  momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            loss_list = []
            for epoch in range(self.dl_studio.epochs):
                running_loss = 0.0
                for i, batch in enumerate(self.train_dataloader):
                    images = batch['image'].to(self.dl_studio.device)
                    masks  = batch['mask_tensor'].to(self.dl_studio.device)
                    # bbox_tensor is available but not used in loss computation.
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    # Log every 500 iterations.
                    if (i + 1) % 500 == 0:
                        elapsed = time.perf_counter() - start_time
                        avg_loss = running_loss / 500.0
                        print(f"[Epoch {epoch+1}/{self.dl_studio.epochs}, Iter {i+1}, Elapsed {int(elapsed)} secs] Loss: {avg_loss:.3f}")
                        fout.write(f"{avg_loss:.3f}\n")
                        fout.flush()
                        loss_list.append(avg_loss)
                        running_loss = 0.0
            # Plot the training loss curve.
            iterations = [500 * i for i in range(1, len(loss_list) + 1)]
            plt.plot(iterations, loss_list, marker='o', label='Training Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title(f"Training Loss Curve; Dice Scale = {dice_scale}")
            plt.legend()
            plt.show()
            print("\nFinished Training\n")
            self.save_model(model)

    def save_model(self, model):
        torch.save(model.state_dict(), self.dl_studio.path_saved_model)
        print(f"Model saved to {self.dl_studio.path_saved_model}")

    def test_segmentation(self, net):
        """
        Test the segmentation network and display outputs.
        """
        net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
        net = net.to(self.dl_studio.device)
        bs = self.dl_studio.batch_size
        H, W = self.dl_studio.image_size
        max_objects = self.max_num_objects
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                images = batch['image'].to(self.dl_studio.device)
                masks  = batch['mask_tensor'].to(self.dl_studio.device)
                outputs = net(images)
                # Display output for every 50th batch.
                if i % 50 == 0:
                    print(f"\nDisplaying output for test batch {i+1}:")
                    # Create a 1-channel output by taking the maximum over the segmentation channels.
                    bw_output = torch.max(outputs, dim=1, keepdim=True)[0]
                    # Stack the outputs, masks, and original images for display.
                    display_tensor = torch.cat([outputs, masks.unsqueeze(1), images], dim=0)
                    grid = torchvision.utils.make_grid(
                        display_tensor, nrow=bs, normalize=True, padding=2, pad_value=10
                    )
                    self.dl_studio.display_tensor_as_image(grid)


# =============================================================================
# Module-level test code (if desired)
# =============================================================================
if __name__ == "__main__":
    # This module is intended to be imported. Add any testing code here if needed.
    pass
