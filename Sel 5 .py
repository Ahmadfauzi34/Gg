# =============================================================================
# CELL 5: MERA Tensor Network Coarse-Graining
# =============================================================================
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class MERALevel:
    disentanglers: List[np.ndarray] = field(default_factory=list)
    isometries: List[np.ndarray] = field(default_factory=list)

class MERAHierarchy:
    def __init__(self, engine, block_size: int = 64):
        self.engine = engine
        self.block_size = block_size
        assert engine.dim % block_size == 0, "dim must be divisible by block_size"
        self.n_blocks = engine.dim // block_size
        self.levels: List[MERALevel] = []

    def _reshape_blocks(self, vec: np.ndarray) -> np.ndarray:
        return np.exp(1j * vec).reshape(self.n_blocks, self.block_size)

    def _flatten_blocks(self, mat: np.ndarray) -> np.ndarray:
        return np.angle(mat).flatten()

    def disentangle_pair(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Za = self._reshape_blocks(a)
        Zb = self._reshape_blocks(b)
        corr = Za @ Zb.conj().T
        from scipy.linalg import svd
        U, s, Vh = svd(corr, full_matrices=False)
        disentangler = U @ Vh
        Za_prime = disentangler @ Za
        Zb_prime = disentangler.conj().T @ Zb
        return self._flatten_blocks(Za_prime), self._flatten_blocks(Zb_prime), s

    def isometric_compress(self, local: np.ndarray, entangled: np.ndarray,
                           spectrum: np.ndarray, retain_ratio: float = 0.5) -> np.ndarray:
        p = spectrum / (spectrum.sum() + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        weight_ent = np.exp(-entropy)
        weight_loc = 1.0 - weight_ent
        w = np.array([weight_loc, weight_ent])
        w = w / w.sum()
        return self.engine.bundle([local, entangled], weights=w)

    def ascend(self, vectors: List[np.ndarray]) -> Tuple[np.ndarray, List[MERALevel]]:
        current = list(vectors)
        self.levels = []
        while len(current) > 1:
            level = MERALevel()
            nxt = []
            if len(current) % 2 == 1:
                current.append(current[-1].copy())
            for i in range(0, len(current), 2):
                a, b = current[i], current[i + 1]
                a_loc, b_ent, spec = self.disentangle_pair(a, b)
                comp = self.isometric_compress(a_loc, b_ent, spec)
                nxt.append(comp)
                level.disentanglers.append(spec)
            self.levels.append(level)
            current = nxt
        return current[0], self.levels

    def descend(self, top_vec: np.ndarray, n_leaves: int,
                levels: Optional[List[MERALevel]] = None) -> List[np.ndarray]:
        if levels is None:
            levels = self.levels
        current = [top_vec]
        for lvl in reversed(levels):
            nxt = []
            for comp in current:
                noise = np.random.normal(0, 0.05, self.engine.dim)
                left = (comp + noise) % (2 * np.pi)
                right = (comp - noise) % (2 * np.pi)
                nxt.extend([left, right])
            current = nxt[:n_leaves * 2]
        return current[:n_leaves]
