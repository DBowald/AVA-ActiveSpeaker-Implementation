import torch
import torch.nn as nn
import torch.optim as optim
from model import ActiveSpeakerModel, ActiveSpeakerTower
from ActiveSpeakerDataset import ActiveSpeakerDataset, ActiveSpeakerCachedDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import datetime
from tqdm import tqdm, trange

softmax = torch.nn.Softmax(dim=1)

class Logger():
    def __init__(self, writer, mode, data_loader=None, net=None, device=None, binary_out=False):
        self.writer = writer
        self.mode = mode
        self.static_images = []
        if data_loader is not None:
            for batch in data_loader:
                self.static_images.append(batch)
                grid = make_grid(batch[0][0], nrow=15)
                self.writer.add_image("{}/static_image_{}".format(self.mode, batch[2][0]), grid)
            self.net = net
            self.device = device
            self.binary_out = binary_out

    def write_loss_acc(self, CMA_loss, CMA_acc, global_step):
        self.writer.add_scalars("Loss", {self.mode: CMA_loss}, global_step)
        self.writer.add_scalars("Accuracy", {self.mode: CMA_acc}, global_step)

    def eval_static_images(self, global_step):
        with torch.no_grad():
            for i,batch in enumerate(self.static_images):
                inp, target, names = batch
                inp, target = inp.to(self.device), target.to(self.device)
                if (self.binary_out):
                    target = torch.Tensor.long(target.mode(keepdim=True)[0]).squeeze(1)
                out = self.net(inp)
                pred_string = names[0] + ": " + str(out[0]).replace("tensor(", "").replace(")", "")
                label_string = names[0] + ": " + str(target[0]).replace("tensor(", "").replace(")", "")
                self.writer.add_text("Pred/{}/static{}".format(self.mode, i), pred_string, global_step)
                self.writer.add_text("Labels/{}/static{}".format(self.mode, i), label_string, global_step)


def main():
    M = 1
    batch_size = 1
    num_workers = 4
    num_epochs = 200
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    binary_out = True
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=1)

    writer = SummaryWriter("./runs/" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))

    #net = ActiveSpeakerModel(M, binary_out=binary_out).to(device)
    net = ActiveSpeakerTower(M).to(device)
    print(net)
    #crit = nn.BCEWithLogitsLoss()
    #crit = nn.MSELoss()
    crit = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
    #optimizer = optim.Adagrad(net.parameters(), lr=2**(-6))
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    #train_dset = ActiveSpeakerDataset("./data/ava_activespeaker_samples/test")
    train_dset = ActiveSpeakerCachedDataset("./data/ava_activespeaker_samples/test_cache")
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dset = ActiveSpeakerCachedDataset("./data/ava_activespeaker_samples/test_cache")
    #val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dset, batch_size, sampler=RandomSampler(val_dset, replacement=True, num_samples=500))

    examples_train_loader = DataLoader(train_dset, batch_size=1, sampler=RandomSampler(train_dset, replacement=True, num_samples=1))
    examples_val_loader = DataLoader(val_dset, batch_size=1, sampler=RandomSampler(val_dset, replacement=True, num_samples=1))
    train_logger = Logger(writer, "train", examples_train_loader, net, device, binary_out)
    val_logger = Logger(writer, "val", examples_val_loader, net, device, binary_out)

    for epoch in range(num_epochs):
        train_loader_tqdm = tqdm(train_loader)
        train_loader_tqdm.set_description("Training Epoch {}/{}".format(epoch, num_epochs))
        train_loss, train_acc = train_or_val("train", train_loader_tqdm, net, device, crit, optimizer, binary_out=binary_out)
        train_logger.write_loss_acc(train_loss, train_acc, epoch)
        train_logger.eval_static_images(epoch)
        val_loader_tqdm = tqdm(val_loader)
        val_loader_tqdm.set_description("Validating Epoch {}/{}".format(epoch, num_epochs))
        val_loss, val_acc = train_or_val("val", val_loader_tqdm, net, device, crit, binary_out=binary_out)
        scheduler.step(val_loss)
        val_logger.write_loss_acc(val_loss, val_acc, epoch)
        val_logger.eval_static_images(epoch)
        # VALIDATION

        #NOTE: SHOULD BE IN VAL
        #scheduler.step(CMA_loss)


def train_or_val(mode, data_loader_tqdm, net, device, crit, optimizer=None, binary_out=False):
    CMA_loss = 0
    CMA_acc = 0
    if mode == "val":
        torch.set_grad_enabled(False)

    for i, batch in enumerate(data_loader_tqdm):
        inp, target, names = batch
        inp, target = inp.to(device), target.to(device)
        if (binary_out):
            target = torch.Tensor.long(target.mode(keepdim=True)[0]).squeeze(1)

        if mode == "train":
            optimizer.zero_grad()
        out = net(inp)
        loss = crit(out, target)

        if mode == "train":
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            global softmax
            acc = torch.mean((target == softmax(out.data).max(1)[1]).float()).item()
            # acc = torch.mean((target == torch.round(sigmoid(out.data))).float()).item()
            # acc = torch.mean((target == torch.round(out.data)).float()).item()

        CMA_loss = CMA_loss + (loss.item() - CMA_loss) / (i + 1)
        CMA_acc = CMA_acc + (acc - CMA_acc) / (i + 1)
        data_loader_tqdm.set_postfix(loss=CMA_loss, acc=CMA_acc)
        # if i % tenth_of_sample == 0:

    if mode == "val":
        torch.set_grad_enabled(True)

    return CMA_loss, CMA_acc
if __name__ == "__main__":
    main()