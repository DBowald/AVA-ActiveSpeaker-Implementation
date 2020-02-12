import torch
import torch.nn as nn
import torch.optim as optim
from model import ActiveSpeakerModel
from ActiveSpeakerDataset import ActiveSpeakerDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import datetime
from tqdm import tqdm, trange

def main():
    M = 15
    batch_size = 4
    num_workers = 8
    num_epochs = 10
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    writer = SummaryWriter("./runs/" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))

    net = ActiveSpeakerModel(M).to(device)
    print(net)
    crit = nn.BCELoss()
    optimizer = optim.Adagrad(net.parameters(), lr=2**(-6))

    train_dset = ActiveSpeakerDataset("./data/ava_activespeaker_samples/train")
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #test_dset = ActiveSpeakerDataset("./data/ava_activespeaker_samples/test")
    #test_loader = DataLoader(test_dset, batch_size, sampler=RandomSampler(test_dset, replacement=True, num_samples=500))
    #epoch_bar = tqdm(range(num_epochs))

    for epoch in trange(num_epochs, desc="Total Progress"):
        CMA_loss = 0
        CMA_acc = 0
        step = 0
        tenth_of_sample = int(len(train_loader)/10)
        pbar = tqdm(train_loader)
        pbar.set_description("Training Epoch {}/{}".format(epoch, num_epochs))
        for i, batch in enumerate(pbar):
            inp, target, names = batch
            if(inp[0].shape[0] != M):
                print("Sample {} does not have {} frames")
                print("Skipping sample")
                continue

            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad()

            out = net(inp)
            loss = crit(out, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = torch.mean((target == torch.round(out.data)).float()).item()

            CMA_loss = CMA_loss + (loss.item() - CMA_loss)/(i+1)
            CMA_acc = CMA_acc + (acc - CMA_acc)/(i+1)
            pbar.set_postfix(loss=CMA_loss, acc=CMA_acc)
            if i % tenth_of_sample == 0:
                global_step = epoch*10 + step
                grid = make_grid(batch[0][0], nrow=15)
                writer.add_image('Examples/train', grid, global_step)
                writer.add_scalar("Loss/train", CMA_loss, global_step)
                pred_string = str(out[0]).replace("tensor(", "").replace(")", "")
                label_string = str(target[0]).replace("tensor(", "").replace(")", "")
                writer.add_text("Pred/train", pred_string, global_step)
                writer.add_text("Labels/train", pred_string, global_step)
                step += 1
                if(i > 0):
                    return

if __name__ == "__main__":
    main()