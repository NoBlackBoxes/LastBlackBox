import torch
import torch.nn as nn

class custom_loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

#FIN