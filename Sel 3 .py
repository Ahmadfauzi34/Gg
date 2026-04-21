# =============================================================================
# CELL 3: Sheaf over Semantic Category Poset
# =============================================================================
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass

@dataclass
class Stalk:
    category: str
    indices: List[int]
    center: np.ndarray
    principal: np.ndarray
    eigenvalues: np.ndarray

class SheafVSA:
    def __init__(self, engine):
        self.engine = engine
        self.stalks: Dict[str, Stalk] = {}
        self.restriction: Dict[Tuple[str, str], np.ndarray] = {}
        self.base_adj: Dict[str, Set[str]] = defaultdict(set)

    def build_stalks(self):
        cats = sorted(set(self.engine.token_categories))
        for cat in cats:
            idxs = [i for i, c in enumerate(self.engine.token_categories) if c == cat]
            vecs = np.stack([self.engine.token_phases[i] for i in idxs])
            center = self.engine.bundle(list(vecs))
            centered = np.exp(1j * (vecs - center))
            cov = centered @ centered.conj().T
            e, v = np.linalg.eigh(cov)
            k = min(5, len(idxs))
            self.stalks[cat] = Stalk(category=cat, indices=idxs, center=center,
                                     principal=v[:, -k:], eigenvalues=e[-k:])

    def build_base_space(self, cooccurrence_threshold: float = 0.15):
        cats = list(self.stalks.keys())
        n = len(cats)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = cats[i], cats[j]
                sim = self.engine.sim(self.stalks[c1].center, self.stalks[c2].center)
                if sim > cooccurrence_threshold:
                    self.base_adj[c1].add(c2)
                    self.base_adj[c2].add(c1)

    def compute_restriction(self, cat1: str, cat2: str) -> np.ndarray:
        c1 = self.stalks[cat1].center
        c2 = self.stalks[cat2].center
        transport = (c2 - c1 + np.pi) % (2 * np.pi) - np.pi
        self.restriction[(cat1, cat2)] = transport
        return transport

    def compute_all_restrictions(self):
        for c1 in self.base_adj:
            for c2 in self.base_adj[c1]:
                if (c1, c2) not in self.restriction:
                    self.compute_restriction(c1, c2)

    def restrict(self, vec: np.ndarray, cat1: str, cat2: str) -> np.ndarray:
        if (cat1, cat2) not in self.restriction:
            self.compute_restriction(cat1, cat2)
        return (vec + self.restriction[(cat1, cat2)]) % (2 * np.pi)

    def global_section_consistency(self, assignment: Dict[str, np.ndarray], tol: float = 0.35):
        violations = []
        for c1 in self.base_adj:
            for c2 in self.base_adj[c1]:
                if c1 not in assignment or c2 not in assignment:
                    continue
                v1r = self.restrict(assignment[c1], c1, c2)
                sim = self.engine.sim(v1r, assignment[c2])
                if sim < tol:
                    violations.append({'edge': (c1, c2), 'similarity': float(sim), 'severity': 1.0 - sim})
        return len(violations) == 0, violations

    def sheaf_cohomology_h0_estimate(self) -> int:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        n_cats = len(self.stalks)
        cats = list(self.stalks.keys())
        idx_map = {c: i for i, c in enumerate(cats)}
        adj = np.zeros((n_cats, n_cats))
        for c1 in self.base_adj:
            for c2 in self.base_adj[c1]:
                if (c1, c2) in self.restriction:
                    trans = self.restriction[(c1, c2)]
                    coherence = float(np.mean(np.cos(trans)))
                    if coherence > 0.5:
                        i, j = idx_map[c1], idx_map[c2]
                        adj[i, j] = 1
        n_comp, _ = connected_components(csgraph=csr_matrix(adj), directed=False)
        return int(n_comp)
