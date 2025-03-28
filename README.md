# serial-neighbor

A lightweight and fast implementation of spatial neighbor search using serialization encoding from space filling curves (e.g., Z-order, Hilbert curves).  
Designed for 3D point cloud applications, fully compatible with PyTorch.

---

## 🚀 Installation

```bash
pip install serial-neighbor
```

---

## 🧠 Function Description

### `serial_neighbor(...)`

Finds k nearest neighbors for each query point using serial encoding. We propose to retrieve a neighborhood from a 1-D ordered list, by serializing points along a space-filling curve, and excluding the impact of points distant from the query (i.e. remove false positives).

<p align="center">
  <img src="assets/Neighboring.png" width="800"/>
</p>



#### **Arguments:**

- `points` (Tensor):  
  Tensor of shape **(M, 3)** giving the source point cloud with M points in 3D space.

- `query_xyz` (Tensor):  
  Tensor of shape **(N, 3)** giving the query points (N points in 3D space).

- `serial_orders` (List[str]):  
  Serialization orders to use, e.g., `["z"]`, `["z", "hilbert"]`, etc.  
  Using multiple orders can improve neighbor search accuracy.

- `k_neighbors` (int):  
  Number of nearest neighbors to find for each query point.

- `grid_size` (float, optional):  
  Grid cell size for point discretization.  
  Smaller values give more accurate results but slower performance. Default: `0.01`

- `mask_threshold` (float, optional):  
  Maximum distance threshold. Neighbors farther than this will be masked with `-1`.  
  Default: `None`

#### **Returns:**

- `combined_idx` (LongTensor):  
  Shape: **(N, K * O)**  
  Indices of the nearest neighbors for each query point.  
  O is the number of serial orders used. Invalid neighbors are marked as `-1`.

- `neighbor_dists` (Tensor):  
  Shape: **(N, K * O)**  
  Euclidean distances to the neighbors.

---

## 🧪 Usage Example

```python
import torch
from serial_neighbor import serial_neighbor

points = torch.rand(1000, 3).cuda()
query_xyz = torch.rand(100, 3).cuda()
idx, dists = serial_neighbor(points, query_xyz, ["z"], k_neighbors=8)

print("Neighbor indices:", idx.shape)
print("Neighbor distances:", dists.shape)
```

---

## 📄 Based on NoKSR

This module is part of the work described in the paper:

**NoKSR: Kernel-Free Neural Surface Reconstruction via Point Cloud Serialization**  
*Zhen Li*, *Weiwei Sun* †, Shrisudhan Govindarajan, Shaobo Xia, Daniel Rebain, Kwang Moo Yi, Andrea Tagliasacchi  
📄 [Paper](https://arxiv.org/abs/2502.12534) | 🔗 [Project Page](https://github.com/theialab/noksr)

> We present a novel approach to large-scale point cloud surface reconstruction by converting an irregular point cloud into a signed distance field (SDF) through serialization-based neighbor search. This framework achieves state-of-the-art performance in both accuracy and efficiency, particularly on large-scale outdoor datasets.

---

## 📰 News

- **[2025/03/22]** The package [`serial-neighbor`](https://pypi.org/project/serial-neighbor/) is released.
- **[2025/02/21]** Code released!
- **[2025/02/19]** ArXiv version is released.

---

## 📬 Contact

For questions, comments, or bug reports, please contact:

**Zhen Li** (SFU) – zhen_li@sfu.ca

---

## 🛡 License

This project is licensed under the **Apache License 2.0** – see the [LICENSE](LICENSE) file for details.
```
