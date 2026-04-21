# =============================================================================
# CELL 6: Spectral Geometry & Diffusion Maps
# =============================================================================
import numpy as np
from scipy.linalg import eigh
from collections import defaultdict
from typing import Dict, List, Optional

class SpectralGeometry:
    def __init__(self, engine):
        self.engine = engine
        self.affinity: Optional[np.ndarray] = None
        self.laplacian: Optional[np.ndarray] = None
        self.laplacian_norm: Optional[np.ndarray] = None
        self.eigvals: Optional[np.ndarray] = None
        self.eigvecs: Optional[np.ndarray] = None

    def build_affinity(self, sigma: float = 0.15, k: int = 20) -> np.ndarray:
        n = len(self.engine.token_names)
        aff = np.zeros((n, n))
        for i in range(n):
            sims = np.array([
                self.engine.sim(self.engine.token_phases[i], self.engine.token_phases[j])
                for j in range(n)
            ])
            knn_thresh = np.partition(sims, -k)[-k] if n > k else -1.0
            for j in range(n):
                if sims[j] >= knn_thresh:
                    aff[i, j] = np.exp(-((1.0 - sims[j]) ** 2) / (2 * sigma ** 2))
        self.affinity = np.maximum(aff, aff.T)
        return self.affinity

    def compute_laplacian(self, normalization: str = 'symmetric'):
        if self.affinity is None:
            self.build_affinity()
        A = self.affinity
        D = np.diag(A.sum(axis=1))
        self.laplacian = D - A
        if normalization == 'symmetric':
            d_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-12))
            self.laplacian_norm = d_inv_sqrt @ self.laplacian @ d_inv_sqrt
        return self.laplacian_norm

    def spectral_embedding(self, n_components: int = 20) -> np.ndarray:
        if self.laplacian_norm is None:
            self.compute_laplacian()
        e, v = eigh(self.laplacian_norm)
        self.eigvals = e
        self.eigvecs = v
        return v[:, 1:n_components + 1]

    def diffusion_map(self, t: float = 2.0, n_components: int = 10) -> np.ndarray:
        if self.affinity is None:
            self.build_affinity()
        A = self.affinity
        D_inv = np.diag(1.0 / (A.sum(axis=1) + 1e-12))
        P = D_inv @ A
        e, v = np.linalg.eig(P)
        idx = np.argsort(-np.real(e))
        e = np.real(e[idx])
        v = np.real(v[:, idx])
        coords = np.zeros((len(e), n_components))
        for i in range(1, n_components + 1):
            coords[:, i - 1] = (e[i] ** t) * v[:, i]
        return coords

    def cheeger_clustering(self, n_clusters: int = 8) -> Dict[int, List[str]]:
        emb = self.spectral_embedding(n_components=n_clusters)
        centroids = emb[np.random.choice(len(emb), n_clusters, replace=False)]
        labels = np.zeros(len(emb), dtype=int)
        for _ in range(20):
            sims = emb @ centroids.T
            labels = np.argmax(sims, axis=1)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    centroids[k] = emb[mask].mean(axis=0)
        clusters = defaultdict(list)
        for idx, lab in enumerate(labels):
            clusters[int(lab)].append(self.engine.token_names[idx])
        return dict(clusters)
