from .default import encode
import torch

def serial_neighbor(points, query_xyz, serial_orders, k_neighbors, grid_size=0.01, mask_threshold=None):
    """
    Finds k nearest neighbors for each query point using serial encoding.

    Args:
        points: Tensor of shape (M, 3) giving the source point cloud containing M points
            in 3D space.
        query_xyz: Tensor of shape (N, 3) giving the query points to find neighbors for,
            containing N points in 3D space.
        serial_orders: List of strings specifying the serialization orders to use
            (e.g. ["z", "hilbert", 'z-trans', 'hilbert-trans']). Multiple orders can improve neighbor search accuracy.
        k_neighbors: Integer giving the number of nearest neighbors to find for each
            query point.
        grid_size: Float giving the grid cell size for point discretization. Smaller values
            give more accurate results but slower performance. Default: 0.01
        mask_threshold: Optional float giving maximum distance threshold. Neighbors farther
            than this are masked out with index -1. Default: None

    Returns:
        combined_idx: LongTensor of shape (N, K*O) giving the indices of neighbors,
            where O is the number of serial orders used. For each query point, it
            contains indices into the source points tensor. Invalid neighbors
            (those beyond the distance threshold) are marked with -1.

        neighbor_dists: Tensor of shape (N, K*O) giving the squared distances to
            the nearest neighbors
    """
    all_neighbor_idx = []
    for order in serial_orders:
        query_coords = torch.floor(query_xyz/grid_size).to(torch.int)
        source_coords = torch.floor(points/grid_size).to(torch.int)
        depth = int(torch.abs(query_coords).max()).bit_length()

        query_codes = encode(query_coords, torch.zeros(query_coords.shape[0], dtype=torch.int64, device=query_coords.device), depth, order=order)
        source_codes = encode(source_coords, torch.zeros(source_coords.shape[0], dtype=torch.int64, device=source_coords.device), depth, order=order)

        sorted_source_codes, sorted_source_indices = torch.sort(source_codes)
        nearest_right_positions = torch.searchsorted(sorted_source_codes, query_codes, right=True)

        k = int(k_neighbors/2) 
        front_indices = nearest_right_positions.unsqueeze(1) - torch.arange(1, k+1).to(nearest_right_positions.device).unsqueeze(0)
        back_indices = nearest_right_positions.unsqueeze(1) + torch.arange(0, k).to(nearest_right_positions.device).unsqueeze(0)

        neighbor_indices = torch.cat((front_indices, back_indices), dim=1)
        neighbor_indices = torch.where((neighbor_indices >= 0) & (neighbor_indices < len(sorted_source_codes)), neighbor_indices, torch.tensor(-1))
        neighbor_idx = torch.where(neighbor_indices != -1, sorted_source_indices[neighbor_indices], torch.tensor(-1))

        all_neighbor_idx.append(neighbor_idx)

    combined_idx = torch.cat(all_neighbor_idx, dim=-1)
    expanded_queries = query_xyz.unsqueeze(1).expand(-1, combined_idx.shape[1], -1)
    neighbor_dists = torch.norm(points[combined_idx] - expanded_queries, dim=-1)

    if mask_threshold is not None:
        dist_mask = neighbor_dists > mask_threshold
        combined_idx[dist_mask] = -1

    return combined_idx, neighbor_dists
