import torch
import torch.nn as nn

"""
将center_loss做成一个网络模块，
以便于可以用单独的优化器去优化中心点参数
"""


class CenterLoss(nn.Module):
    def __init__(self, label_num, feature_num):
        super(CenterLoss, self).__init__()
        self.center = nn.Parameter(torch.randn(label_num, feature_num), requires_grad=True)

    def forward(self, feature, labels, lamda=2):
        center_exp = self.center.index_select(dim=0, index=labels.long())

        count = torch.histc(labels, bins=int(max(labels).item() + 1), min=0, max=max(labels).item())
        count_exp = count.index_select(dim=0, index=labels.long())

        loss = lamda / 2 * torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp))
        return loss, self.center


if __name__ == '__main__':

    data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32).cuda()
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32).cuda()
    center_loss = CenterLoss(10, 2).cuda()
    loss = center_loss(data, label, 2)
    print(loss)