from __future__ import annotations

import numpy as np

from scipy.spatial import cKDTree


def sanitize_pixel_coords(coords, pixel_size):
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Invalid coords shape: {arr.shape}, expected (n_features, 2)")
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr)
    arr = arr.astype(np.int64, copy=False)
    height, width = (int(pixel_size[0]), int(pixel_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid pixel_size: {pixel_size}")
    arr[:, 0] = np.clip(arr[:, 0], 0, height - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, width - 1)
    return arr


def balanced_greedy_kdtree_assign_coords(
    bin_coords, pixel_size, patch_size=16, occupied_mask=None
):
    """Assign coordinates while spreading genes across image patches.

    This keeps the same nearest-pixel spirit as ``greedy_kdtree_assign_coords``:
    for each target coordinate, candidates are searched by KD-tree distance. When
    several candidates are available, the pixel from the least occupied patch is
    preferred. If there are more genes than pixels, every pixel is used once
    before controlled overlap is allowed.
    """
    if bin_coords is None or len(bin_coords) == 0:
        return None
    coords = np.asarray(bin_coords, dtype=float)
    height, width = (int(pixel_size[0]), int(pixel_size[1]))
    if height <= 0 or width <= 0:
        return None
    patch_size = max(1, int(patch_size))
    targets = np.rint(coords).astype(int)
    targets[:, 0] = np.clip(targets[:, 0], 0, height - 1)
    targets[:, 1] = np.clip(targets[:, 1], 0, width - 1)
    rr, cc = np.indices((height, width))
    pixel_coords = np.column_stack((rr.ravel(), cc.ravel()))
    n_pixels = pixel_coords.shape[0]
    if n_pixels == 0:
        return None
    patch_rows = (height + patch_size - 1) // patch_size
    patch_cols = (width + patch_size - 1) // patch_size
    pixel_patch_ids = (
        pixel_coords[:, 0] // patch_size * patch_cols + pixel_coords[:, 1] // patch_size
    )
    patch_counts = np.zeros(patch_rows * patch_cols, dtype=np.int64)
    pixel_counts = np.zeros(n_pixels, dtype=np.int64)
    used_once = np.zeros(n_pixels, dtype=bool)
    tree = cKDTree(pixel_coords)
    assigned = np.empty((targets.shape[0], 2), dtype=int)
    if occupied_mask is not None:
        used_once = np.asarray(occupied_mask, dtype=bool).ravel().copy()
        if used_once.size != n_pixels:
            raise ValueError("occupied_mask has incompatible shape")
    else:
        used_once = np.zeros(n_pixels, dtype=bool)
    available = ~used_once

    def choose(candidates):
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.size == 0:
            return None
        patch_load = patch_counts[pixel_patch_ids[candidates]]
        pix_load = pixel_counts[candidates]
        order = np.lexsort((candidates, pix_load, patch_load))
        return int(candidates[int(order[0])])

    for i, target in enumerate(targets):
        chosen_idx = None
        k = 1
        while True:
            k_now = min(k, n_pixels)
            _, idxs = tree.query(target, k=k_now)
            idxs = np.atleast_1d(idxs).astype(int)
            free_candidates = idxs[available[idxs] & ~used_once[idxs]]
            chosen_idx = choose(free_candidates)
            if chosen_idx is not None:
                break
            if k_now == n_pixels:
                break
            k = min(k_now * 2, n_pixels)
        if chosen_idx is None:
            k = 1
            while True:
                k_now = min(k, n_pixels)
                _, idxs = tree.query(target, k=k_now)
                idxs = np.atleast_1d(idxs).astype(int)
                chosen_idx = choose(idxs[available[idxs]])
                if chosen_idx is not None or k_now == n_pixels:
                    break
                k = min(k_now * 2, n_pixels)
        if chosen_idx is None:
            chosen_idx = 0
        used_once[chosen_idx] = True
        pixel_counts[chosen_idx] += 1
        patch_counts[pixel_patch_ids[chosen_idx]] += 1
        assigned[i] = pixel_coords[chosen_idx]
    return sanitize_pixel_coords(assigned, pixel_size)
