import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import glob
import os

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


if __name__ == "__main__":
    dl = DataLoader(ActiveSpeakerDataset("./data/ava_activespeaker_samples/train"), shuffle=True, batch_size=4)
    example = next(iter(dl))
    print(example[0].shape)
    print(example[0].squeeze().shape)
    batch = example[0][0]
    grid = torchvision.utils.make_grid(batch, nrow=15)
    to_im = transforms.Compose([transforms.ToPILImage()])
    to_im(grid).show()
    print(example[1][0])
    print(example[2][0])