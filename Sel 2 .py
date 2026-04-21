# =============================================================================
# CELL 2: Vietoris-Rips Persistent Homology (Z2)
# =============================================================================
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass

@dataclass
class PersistencePair:
    dim: int
    birth: float
    death: float
    creator: Tuple[int, ...] = ()
    destroyer: Tuple[int, ...] = ()

class VietorisRipsPH:
    def __init__(self, max_dim: int = 1, k_neighbors: int = 12, max_dist: float = 1.5):
        self.max_dim = max_dim
        self.k = k_neighbors
        self.max_dist = max_dist
        self.simplices: List[Tuple[float, int, Tuple[int, ...]]] = []
        self.pairs: List[PersistencePair] = []
        self.positive: List[PersistencePair] = []
        self._index_map: Dict[Tuple[int, ...], int] = {}
        self._reduced: List[Set[int]] = []

    def fit(self, dist_matrix: np.ndarray):
        n = dist_matrix.shape[0]
        neighbors = self._knn_graph(dist_matrix, n)
        self._enumerate_simplices(dist_matrix, neighbors, n)
        self._reduce_boundary_matrix()
        return self

    def _knn_graph(self, D: np.ndarray, n: int) -> List[np.ndarray]:
        neigh = []
        for i in range(n):
            idx = np.argsort(D[i])[:self.k + 1]
            idx = idx[idx != i][:self.k]
            neigh.append(idx)
        return neigh

    def _enumerate_simplices(self, D: np.ndarray, neighbors: List[np.ndarray], n: int):
        simplices = []
        for i in range(n):
            simplices.append((0.0, 0, (i,)))
        edges = set()
        for i in range(n):
            for j in neighbors[i]:
                if i < j:
                    d = D[i, j]
                    if d <= self.max_dist:
                        edges.add((i, j))
                        simplices.append((float(d), 1, (i, j)))
        if self.max_dim >= 2:
            for i in range(n):
                nei = list(neighbors[i])
                for a in range(len(nei)):
                    for b in range(a + 1, len(nei)):
                        j, k = nei[a], nei[b]
                        if i < j and i < k and (j, k) in edges:
                            d = max(D[i,j], D[i,k], D[j,k])
                            if d <= self.max_dist:
                                simplices.append((float(d), 2, (i, j, k)))
        simplices.sort(key=lambda x: (x[0], x[1]))
        self.simplices = simplices
        self._index_map = {simp: idx for idx, (_, _, simp) in enumerate(simplices)}

    def _boundary(self, simplex: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        return [simplex[:i] + simplex[i+1:] for i in range(len(simplex))]

    def _reduce_boundary_matrix(self):
        cols = []
        for _, dim, simp in self.simplices:
            if dim == 0:
                cols.append(set())
            else:
                bdry = set()
                for face in self._boundary(simp):
                    if face in self._index_map:
                        bdry.add(self._index_map[face])
                cols.append(bdry)
        reduced = [set(c) for c in cols]
        low = {}
        self.pairs = []
        self.positive = []
        for j in range(len(reduced)):
            col = reduced[j]
            while col:
                l = max(col)
                if l in low:
                    i = low[l]
                    col = col.symmetric_difference(reduced[i])
                else:
                    low[l] = j
                    birth_val, dim_l, creator = self.simplices[l]
                    death_val, _, destroyer = self.simplices[j]
                    self.pairs.append(PersistencePair(dim=dim_l, birth=birth_val, death=death_val,
                                                      creator=creator, destroyer=destroyer))
                    break
            if not col:
                birth_val, dim_j, creator = self.simplices[j]
                self.positive.append(PersistencePair(dim=dim_j, birth=birth_val, death=float('inf'),
                                                     creator=creator))
        self._reduced = reduced

    def betti(self, threshold: float) -> Dict[int, int]:
        alive = defaultdict(int)
        for p in self.pairs:
            if p.birth <= threshold < p.death:
                alive[p.dim] += 1
        for p in self.positive:
            if p.birth <= threshold:
                alive[p.dim] += 1
        return dict(alive)

    def persistence_diagram(self):
        births, deaths, dims = [], [], []
        for p in self.pairs + self.positive:
            births.append(p.birth)
            deaths.append(p.death if p.death != float('inf') else max(births) * 1.1 if births else 1.0)
            dims.append(p.dim)
        return np.array(births), np.array(deaths), np.array(dims)

    def significant_features(self, min_persistence: float = 0.1) -> List[PersistencePair]:
        out = []
        for p in self.pairs:
            if p.death - p.birth > min_persistence:
                out.append(p)
        for p in self.positive:
            if p.death == float('inf'):
                out.append(p)
        return out
