import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),  # 28*28
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),  # 28*28
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.MaxPool2d(2, 2),  # 14*14
            nn.Conv2d(32, 64, 5, 1, 2),  # 14*14
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),  # 14*14
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.MaxPool2d(2, 2),  # 7*7
            nn.Conv2d(64, 128, 5, 1, 2),  # 7*7
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 5, 1, 2),  # 7*7
            nn.BatchNorm2d(128),
            nn.PReLU(),

            nn.MaxPool2d(2, 2)  # 3*3
        )

        self.feature = nn.Linear(128 * 3 * 3, 2)
        self.output = nn.Linear(2, 10)

    def forward(self, x):
        y_conv = self.conv_layer(x)
        y_conv = torch.reshape(y_conv, [-1, 128 * 3 * 3])
        y_feature = self.feature(y_conv)  # [n,2]
        y_output = torch.log_softmax(self.output(y_feature), dim=1)
        return y_feature, y_output  # [n,10]

    # 将特征点[60000,2]和标签[60000,]传入画图并保存图片
    def visualize(self, feat, labels, epoch, center_point):
        color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
                 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.clf()
        for i in range(10):
            plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.plot(center_point[:, 0], center_point[:, 1], 'o', c='gray')
        plt.title("epoch={}".format(epoch))
        plt.savefig('./images/epoch={}.jpg'.format(epoch))


if __name__ == '__main__':
    x = torch.randn(20, 1, 28, 28)
    net = Net()
    feature, output = net(x)
    print(feature.shape)
    print(output.shape)
