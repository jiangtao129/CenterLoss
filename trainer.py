import torch
import torchvision
import torchvision.transforms as trans
from torch.utils.data import DataLoader
import torch.nn as nn
from CenterLossImprove.net import Net
from CenterLossImprove.center_loss import CenterLoss
import os


class Trainer:
    def __init__(self):
        self.save_cls = r"./models/cls_net.pth"
        self.save_center = r"./models/center_net.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(mean=[0.5, ], std=[0.5, ])
        ])
        self.data = torchvision.datasets.MNIST("./MNIST", train=True, transform=transform, download=True)
        self.dataloder = DataLoader(self.data, batch_size=100, shuffle=True, num_workers=4)
        self.net = Net().to(self.device)
        self.cls_loss_func = nn.NLLLoss().to(self.device)
        self.center_func = CenterLoss(10, 2).to(self.device)

        self.cls_optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.center_optimizer = torch.optim.SGD(self.center_func.parameters(), lr=0.5)

        if os.path.exists(self.save_cls):
            self.net.load_state_dict(torch.load(self.save_cls))
        if os.path.exists(self.save_center):
            self.center_func.load_state_dict(torch.load(self.save_center))

    def train(self):
        epochs = 0
        while True:
            label_loder = []
            feature_loder = []
            for i, (x, y) in enumerate(self.dataloder):
                x = x.to(self.device)
                y = y.to(self.device)
                out_feature, output = self.net(x)
                cls_loss = self.cls_loss_func(output, y)
                center_loss, center_point = self.center_func(out_feature, y.float())
                loss = cls_loss + center_loss

                label_loder.append(y)
                feature_loder.append(out_feature)

                self.cls_optimizer.zero_grad()
                self.center_optimizer.zero_grad()
                loss.backward()
                self.cls_optimizer.step()
                self.center_optimizer.step()

                if i % 600 == 0:
                    print("epochs:{}, loss:{}, cls_loss:{}, center_loss:{}".format(epochs, loss.item(), cls_loss.item(),
                                                                                   center_loss.item()))

            torch.save(self.net.state_dict(), self.save_cls)
            torch.save(self.center_func.state_dict(), self.save_center)

            feature_cat = torch.cat(feature_loder, dim=0).data.cpu().numpy()
            label_cat = torch.cat(label_loder, dim=0).data.cpu().numpy()
            self.net.visualize(feature_cat, label_cat, epochs, center_point.data.cpu().numpy())

            epochs += 1
            if epochs == 50:
                break


if __name__ == '__main__':
    t = Trainer()
    t.train()
