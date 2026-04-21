# =============================================================================
# CELL 4: Fiber Bundle (Category Base × Phase Fiber)
# =============================================================================
import numpy as np
from typing import Dict, List, Tuple, Optional

class FiberBundleVSA:
    def __init__(self, engine):
        self.engine = engine
        self.fibers: Dict[str, np.ndarray] = {}
        self.connections: Dict[Tuple[str, str], np.ndarray] = {}
        self.base_paths: Dict[Tuple[str, str], List[str]] = {}

    def build_fibers(self):
        cats = sorted(set(self.engine.token_categories))
        for cat in cats:
            idxs = [i for i, c in enumerate(self.engine.token_categories) if c == cat]
            self.fibers[cat] = np.stack([self.engine.token_phases[i] for i in idxs])

    def build_connections(self, sheaf):
        for (c1, c2), transport in sheaf.restriction.items():
            self.connections[(c1, c2)] = transport.copy()

    def parallel_transport(self, vec: np.ndarray, path: List[str]) -> np.ndarray:
        current = vec.copy()
        for i in range(len(path) - 1):
            c1, c2 = path[i], path[i + 1]
            if (c1, c2) in self.connections:
                current = (current + self.connections[(c1, c2)]) % (2 * np.pi)
        return current

    def compute_geodesic(self, cat1: str, cat2: str, sheaf) -> List[str]:
        if not self.base_paths:
            self._precompute_paths(sheaf)
        return self.base_paths.get((cat1, cat2), [cat1, cat2])

    def _precompute_paths(self, sheaf):
        cats = list(sheaf.stalks.keys())
        n = len(cats)
        idx = {c: i for i, c in enumerate(cats)}
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        for c1 in sheaf.base_adj:
            for c2 in sheaf.base_adj[c1]:
                i, j = idx[c1], idx[c2]
                dist[i, j] = 1.0
        pred = np.full((n, n), -1, dtype=int)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        pred[i, j] = k
        def path(i, j):
            if pred[i, j] == -1:
                return [cats[i], cats[j]]
            k = pred[i, j]
            return path(i, k)[:-1] + path(k, j)
        for i in range(n):
            for j in range(n):
                if i != j and dist[i, j] < np.inf:
                    self.base_paths[(cats[i], cats[j])] = path(i, j)

    def curvature(self, loop: List[str], test_vec: np.ndarray) -> float:
        transported = self.parallel_transport(test_vec, loop + [loop[0]])
        diff = (transported - test_vec + np.pi) % (2 * np.pi) - np.pi
        return float(np.mean(np.abs(diff)))

    def section(self, category: str, token_name: Optional[str] = None) -> np.ndarray:
        if token_name is None:
            return self.engine.bundle(list(self.fibers[category]))
        idx = self.engine.token_names.index(token_name)
        return self.engine.token_phases[idx]
