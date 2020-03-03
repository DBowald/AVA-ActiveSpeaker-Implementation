import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import glob
import os
from resnets.spatial_transforms import Compose, Normalize, Scale, CenterCrop, ToTensor

import numpy as np
import cv2
import pandas as pd
import wget

class ActiveSpeakerDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = glob.glob(os.path.join(root_dir, '*'))
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        #frames = os.listdir(sample_path)
        frame_paths = glob.glob(os.path.join(sample_path, "*.png"))
        tensors = []
        outputs = torch.zeros(15)
        for i, each_frame_path in enumerate(frame_paths):
            #im = Image.open(os.path.join(sample_path, each_frame))
            frame = Image.open(each_frame_path)
            tensors.append(self.transforms(frame))
            if "SPEAKING_AUDIBLE" in each_frame_path or "SPEAKING_NOT_AUDIBLE" in each_frame_path:
                outputs[i] = 1.0
            else:
                outputs[i] = 0.0
        return torch.stack(tensors, dim=0), outputs, sample_path

class ActiveSpeakerCachedDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = glob.glob(os.path.join(root_dir, '*'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        frames = torch.load(os.path.join(sample_path, "frames.pt"))
        labels = torch.load(os.path.join(sample_path, "labels.pt"))
        #add = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #frames = add(frames[0])
        #frames = frames

        norm_method = Normalize([114.7748, 107.7354, 99.475], [1, 1, 1])

        back_to_PIL = transforms.Compose([transforms.ToPILImage()])

        spatial_transform = Compose([
            Scale(112),
            CenterCrop(112),
            ToTensor(1)#, norm_method
        ])

        new_frames = []
        for each_frame in frames:
            each_frame = back_to_PIL(each_frame)
            new_frames.append(spatial_transform(each_frame))
        frames = torch.stack(new_frames)

        frames = frames.permute(1,0,2,3)
        return frames, labels, sample_path

if __name__ == "__main__":
    dl = DataLoader(ActiveSpeakerDataset("./data/ava_activespeaker_samples/train"), shuffle=True, batch_size=4)
    example = next(iter(dl))
    print(example[0].shape)
    print(example[0].squeeze().shape)
    batch = example[0][0]
    grid = torchvision.utils.make_grid(batch, nrow=16)
    to_im = transforms.Compose([transforms.ToPILImage()])
    to_im(grid).show()
    print(example[1][0])
    print(example[2][0])