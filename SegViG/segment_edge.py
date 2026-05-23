import torch
import torch.nn.functional as F


def pixel_to_patch_segment(segmentation, patch_num):
    """
    Args:
        segmentation: torch.Tensor [Batch, H, W]
        patch_num: int, number of patches per side (e.g., 14 for 14x14)
    Returns:
        patch_segments: torch.Tensor [Batch, patch_num, patch_num]
    """
    batch_size, h, w = segmentation.shape
    patch_size = h // patch_num
    new_hw = h // patch_size

    patches = segmentation.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.reshape(batch_size, new_hw, new_hw, patch_size * patch_size)
    patch_segments = patches.mode(dim=-1).values
    return patch_segments


def pixel_to_patch_segment_overlapping(segmentation, kernel_size=3, stride=2, padding=1):
    """
    Args:
        segmentation: torch.Tensor [Batch, H, W]
    Returns:
        patch_segments: torch.Tensor [Batch, patch_num, patch_num]
    """
    batch_size, h, w = segmentation.shape
    segmentation = segmentation.unsqueeze(1).float()

    for _ in range(4):
        segmentation = F.unfold(segmentation, kernel_size=kernel_size, stride=stride, padding=padding)
        new_size = (h + 2 * 1 - 3) // 2 + 1
        segmentation = segmentation.view(batch_size, -1, new_size, new_size)
        h, w = new_size, new_size

    patch_segments = segmentation.mode(dim=1).values
    return patch_segments


def segment_based_edge_index(x, segment_labels):
    """
    Build an edge index where nodes in the same segment are connected.
    Pads shorter neighbor lists with self-connections for uniform shape.

    Args:
        x: (batch_size, num_dims, num_points, 1)
        segment_labels: (batch_size, patch_num, patch_num)
    Returns:
        edge_index: (2, batch_size, num_points, max_k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, point_1, _, point_2 = x.shape
        n_points = point_1 * point_2
        segment_labels = torch.reshape(segment_labels, (batch_size, -1))

        max_k_list = []
        edge_index_list = []
        for batch in range(batch_size):
            labels = segment_labels[batch]
            edge_indices_batch = []
            for node_idx in range(n_points):
                node_label = labels[node_idx]
                neighbors = (labels == node_label).nonzero(as_tuple=True)[0]
                edge_indices_batch.append(neighbors)

            max_k = max(len(nb) for nb in edge_indices_batch)
            max_k_list.append(max_k)

            padded_edges = [
                torch.cat([nb, nb.new_full((max_k - len(nb),), nb[0])])
                if len(nb) < max_k else nb
                for nb in edge_indices_batch
            ]
            edge_index_list.append(torch.stack(padded_edges, dim=0))

        global_max_k = max(max_k_list)
        for i in range(len(edge_index_list)):
            batch_edges = edge_index_list[i]
            if batch_edges.shape[1] < global_max_k:
                pad_size = global_max_k - batch_edges.shape[1]
                padding = batch_edges.new_full((n_points, pad_size), batch_edges[0, 0])
                edge_index_list[i] = torch.cat([batch_edges, padding], dim=1)

        edge_index = torch.stack(edge_index_list, dim=0)
        center_idx = (torch.arange(0, n_points, device=x.device)
                      .repeat(batch_size, global_max_k, 1).transpose(2, 1))
    return torch.stack((edge_index, center_idx), dim=0)
