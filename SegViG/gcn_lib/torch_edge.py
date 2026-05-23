import math
import torch
from torch import nn
import torch.nn.functional as F


def pairwise_distance(x):
    """
    Args:
        x: (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2 * torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, k=16, relative_pos=None):
    """
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        edge_index: (2, batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list.append(nn_idx_part)
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k)
        center_idx = (torch.arange(0, n_points, device=x.device)
                      .repeat(batch_size, k, 1).transpose(2, 1))
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = (torch.arange(0, n_points, device=x.device)
                      .repeat(batch_size, k, 1).transpose(2, 1))
    return torch.stack((nn_idx, center_idx), dim=0)


def dense_knn_seg_matrix(x, seg_label, k=16, relative_pos=None):
    """KNN restricted to nodes sharing the same segment label (mask-and-replace approach)."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1, 1).contiguous()
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        seg_label = seg_label.view(batch_size, -1)

        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list.append(nn_idx_part)
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k)

        seg_nn_idx = nn_idx.clone()
        batch_idx = (torch.arange(batch_size, device=nn_idx.device)
                     .view(batch_size, 1, 1).expand(batch_size, n_points, k))
        nn_idx_labels = seg_label[batch_idx, nn_idx]
        seg_labels = seg_label.unsqueeze(-1)
        valid_mask = nn_idx_labels == seg_labels
        pad_value = nn_idx[:, :, 0:1].expand_as(nn_idx)
        seg_nn_idx = torch.where(valid_mask, nn_idx, pad_value)
        center_idx = (torch.arange(0, n_points, device=x.device)
                      .repeat(batch_size, k, 1).transpose(2, 1))
    return torch.stack((seg_nn_idx, center_idx), dim=0)


def dense_knn_seg_matrix_v2(x, seg_label, k=16, relative_pos=None):
    """KNN restricted to same-segment nodes (distance masking approach)."""
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        if len(seg_label.shape) == 3:
            seg_label = seg_label.reshape(batch_size, -1)

        n_part = 10000
        if n_points > n_part:
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i
                end_idx = min(n_points, n_part * (i + 1))
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                seg_row = seg_label.unsqueeze(-1)
                seg_col = seg_label.unsqueeze(1)
                mask = seg_row != seg_col
                dist.masked_fill_(mask, float('inf'))
                _, nn_idx_part = torch.topk(-dist, k=k)
                nn_idx_list.append(nn_idx_part)
            nn_idx = torch.cat(nn_idx_list, dim=1)
        else:
            dist = pairwise_distance(x.detach())
            if relative_pos is not None:
                dist += relative_pos
            seg_row = seg_label.unsqueeze(-1)
            seg_col = seg_label.unsqueeze(1)
            mask = seg_row != seg_col
            dist.masked_fill_(mask, float('inf'))
            _, nn_idx = torch.topk(-dist, k=k)

        center_idx = (torch.arange(0, n_points, device=x.device)
                      .repeat(batch_size, k, 1).transpose(2, 1))
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """Select every `dilation`-th neighbor from the kNN list."""
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """Build dilated kNN graph, optionally restricted by segmentation labels."""
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0, use_segmentation_edge=False):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.use_segmentation_edge = use_segmentation_edge

    def forward(self, x, y=None, relative_pos=None, seg_label=None):
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos)
        elif self.use_segmentation_edge == 'seg_only':
            edge_index = seg_label
        elif self.use_segmentation_edge == 'seg_knn':
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = dense_knn_seg_matrix(x, seg_label, self.k * self.dilation, relative_pos)
        elif self.use_segmentation_edge == 'knn_in_seg':
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = dense_knn_seg_matrix_v2(x, seg_label, self.k * self.dilation, relative_pos)
        elif self.use_segmentation_edge is False:
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)
        else:
            raise ValueError(f'Unknown use_segmentation_edge value: {self.use_segmentation_edge}')
        return self._dilated(edge_index)
