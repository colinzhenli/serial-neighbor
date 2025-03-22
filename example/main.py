import torch
from serial_neighbor import serial_neighbor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_source_points = 1000
    num_query_points = 100
    grid_size = 0.005

    points = torch.rand(num_source_points, 3).to(device)
    query_xyz = torch.rand(num_query_points, 3).to(device)

    serial_orders = ["z"]
    k_neighbors = 8

    combined_idx, neighbor_dists = serial_neighbor(points, query_xyz, serial_orders, k_neighbors, grid_size, mask_threshold=0.01)

    print(f"Input points shape: {points.shape}")
    print(f"Query points shape: {query_xyz.shape}")
    print(f"Neighbor indices shape: {combined_idx.shape}")
    print(f"Neighbor distances shape: {neighbor_dists.shape}")

if __name__ == "__main__":
    main()
