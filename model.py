import torch
import torch.nn.functional as F
import torch.nn as nn

class ActiveSpeakerTower(nn.Module):
    def __init__(self, M):
        super(ActiveSpeakerTower, self).__init__()
        self.conv1 = nn.Conv2d(M, 32, (3,3), stride=2, padding=1)
        self.conv2a = nn.Conv2d(32, 32, (3,3), stride=1, padding=1)#, groups=32)
        self.conv2b = nn.Conv2d(32, 64, (1,1), stride=1, padding=0)#, groups=32)
        self.downsampler = self.get_downsampler(64, 64)
        self.pool = nn.AvgPool2d((2, 2))
        self.fc = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, inp):
        inp = inp.squeeze(2)
        inp = self.conv1(inp)
        inp = self.relu(inp)

        inp = self.conv2a(inp)
        inp = self.conv2b(inp)
        inp = self.relu(inp)

        inp = self.downsampler(inp)

        inp = self.pool(inp)
        inp = self.fc(inp.squeeze(-1).squeeze(-1))
        inp = self.relu(inp)
        return self.fc2(inp)

    def get_downsampler(self, in_c, in_size):
        curr_size = in_size
        layers = []
        while(curr_size > 2):
            layers.append(nn.Conv2d(in_c, in_c, (3,3), stride=2, padding=1))#, groups=in_c))
            curr_size /= 2
            layers.append(nn.Conv2d(in_c, in_c, (1,1), stride=1, padding=0))#, groups=in_c))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

class ActiveSpeakerModel(nn.Module):
    def __init__(self, M, binary_out=False):
        super(ActiveSpeakerModel, self).__init__()
        self.vis_tower = ActiveSpeakerTower(M)
        if(binary_out):
            self.fc = nn.Linear(128, 1)
        else:
            self.fc = nn.Linear(128, M)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        inp = self.vis_tower(inp)
        inp = self.fc(inp)
        #Note: Sigmoid done in BCEWithLogitsLoss
        #inp = self.sigmoid(inp)
        return inp


if __name__ == "__main__":
    model = ActiveSpeakerModel(15)
    fake_batch = torch.rand(4,15,128,128)
    out = model(fake_batch)
    print(out.shape, out)