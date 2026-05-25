from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class FeatureIndex:
    def __init__(self, path: str | Path, device: str | torch.device, dtype: torch.dtype = torch.float32):
        with np.load(path, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"index features must be 2D, got shape={features.shape}")
        self.features = torch.from_numpy(features).to(device=device, dtype=dtype).contiguous()
        self.search_features = self.features if dtype != torch.float32 else self.features.float()
        self.search_features_t = self.search_features.t().contiguous()
        self.squared_norms = self.features.float().pow(2).sum(dim=1)

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def blend(self, query: torch.Tensor, rate: float, top_k: int = 8, query_chunk_size: int = 1024) -> torch.Tensor:
        if rate <= 0:
            return query
        if query.shape[-1] != self.dim:
            raise ValueError(f"index dim {self.dim} does not match query dim {query.shape[-1]}")
        original_dtype = query.dtype
        flat_query = query.squeeze(0).float()
        k = min(int(top_k), int(self.features.shape[0]))
        chunks = [
            self._retrieve_chunk(flat_query[start : start + query_chunk_size], k)
            for start in range(0, flat_query.shape[0], query_chunk_size)
        ]
        retrieved = torch.cat(chunks, dim=0).unsqueeze(0)
        mixed = retrieved.to(query.dtype) * float(rate) + query * float(1.0 - rate)
        return mixed.to(original_dtype)

    def _squared_l2(self, query: torch.Tensor) -> torch.Tensor:
        query_norms = query.pow(2).sum(dim=1, keepdim=True)
        dot = query.to(self.search_features.dtype) @ self.search_features_t
        distances = query_norms + self.squared_norms.unsqueeze(0) - 2.0 * dot.float()
        return distances.clamp_min_(0.0)

    def _retrieve_chunk(self, query: torch.Tensor, k: int) -> torch.Tensor:
        values, indices = torch.topk(self._squared_l2(query), k=k, dim=1, largest=False)
        weights = values.clamp_min(1e-12).reciprocal().square()
        weights = weights / weights.sum(dim=1, keepdim=True)
        return (self.features[indices].float() * weights.unsqueeze(-1)).sum(dim=1)
