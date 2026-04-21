# =============================================================================
# CELL 7: FHRR Topological Layer (Unified API)
# =============================================================================
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class FHRRTopologicalLayer:
    def __init__(self, engine):
        self.engine = engine
        self.ph = VietorisRipsPH(max_dim=1, k_neighbors=12, max_dist=1.2)
        self.sheaf = SheafVSA(engine)
        self.bundle = FiberBundleVSA(engine)
        self.mera = MERAHierarchy(engine, block_size=64)
        self.spectral = SpectralGeometry(engine)
        self._dist_matrix: Optional[np.ndarray] = None
        self._cached_tokens: int = 0

    def _ensure_distance_matrix(self):
        n = len(self.engine.token_names)
        if self._dist_matrix is not None and self._cached_tokens == n:
            return self._dist_matrix
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - self.engine.sim(self.engine.token_phases[i], self.engine.token_phases[j])
                dist[i, j] = d
                dist[j, i] = d
        self._dist_matrix = dist
        self._cached_tokens = n
        return dist

    def analyze_vocabulary_topology(self) -> Dict[str, Any]:
        print("[Topo] Distance matrix...")
        D = self._ensure_distance_matrix()
        print("[Topo] Persistent Homology...")
        self.ph.fit(D)
        betti = self.ph.betti(threshold=0.5)
        print("[Topo] Sheaf & Fiber...")
        self.sheaf.build_stalks()
        self.sheaf.build_base_space()
        self.sheaf.compute_all_restrictions()
        self.bundle.build_fibers()
        self.bundle.build_connections(self.sheaf)
        h0 = self.sheaf.sheaf_cohomology_h0_estimate()
        print("[Topo] Spectral geometry...")
        emb = self.spectral.spectral_embedding(n_components=15)
        clusters = self.spectral.cheeger_clustering(n_clusters=10)
        return {
            'betti_numbers': betti,
            'significant_features': len(self.ph.significant_features(0.1)),
            'sheaf_h0': h0,
            'spectral_clusters': clusters,
            'embedding_shape': emb.shape
        }

    def mera_encode_sentence(self, bindings: Dict[str, str]) -> Tuple[np.ndarray, List[MERALevel]]:
        vectors = []
        for role_name, token_name in bindings.items():
            role_vec = self.engine.get_role(role_name)
            token_vec = self.engine.get_token(token_name)
            if role_vec is None or token_vec is None:
                continue
            bound = self.engine.bind(role_vec, token_vec, out=np.zeros(self.engine.dim))
            vectors.append(bound)
        if not vectors:
            raise ValueError("No valid bindings")
        return self.mera.ascend(vectors)

    def topological_query(self, query_vec: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        match, conf = self.engine.cleanup(query_vec)
        decoded = self.engine.decode(query_vec, threshold=0.35)
        cat_compat = {}
        for role, (tok, c) in decoded.items():
            cat = self.engine.token_categories[self.engine.token_names.index(tok)]
            if cat in self.sheaf.stalks:
                center = self.sheaf.stalks[cat].center
                cat_compat[role] = float(self.engine.sim(query_vec, center))
        kernel_results = self.engine.kernel_query(query_vec, radius=0.25, top_k=top_k)
        return {
            'cleanup_match': match,
            'cleanup_conf': conf,
            'decoded': decoded,
            'category_compatibility': cat_compat,
            'kernel_neighbors': kernel_results
        }

    def detect_topological_contradiction(self, vec1: np.ndarray, vec2: np.ndarray) -> Dict[str, Any]:
        is_contra, conflicts = self.engine.detect_contradiction(vec1, vec2)
        dec1 = self.engine.decode(vec1, threshold=0.35)
        dec2 = self.engine.decode(vec2, threshold=0.35)
        sheaf_conflicts = []
        for role in set(dec1.keys()) & set(dec2.keys()):
            t1, _ = dec1[role]
            t2, _ = dec2[role]
            c1 = self.engine.token_categories[self.engine.token_names.index(t1)]
            c2 = self.engine.token_categories[self.engine.token_names.index(t2)]
            if c1 != c2:
                sheaf_conflicts.append({
                    'role': role, 'token1': t1, 'cat1': c1,
                    'token2': t2, 'cat2': c2
                })
        curvature = None
        if len(self.bundle.connections) > 2:
            cats = list(self.bundle.fibers.keys())[:4]
            loop = cats + [cats[0]]
            curvature = self.bundle.curvature(loop, vec1)
        return {
            'standard_contradiction': is_contra,
            'standard_conflicts': conflicts,
            'sheaf_conflicts': sheaf_conflicts,
            'bundle_curvature': curvature
        }

    def analogy_via_fiber_transport(self, source_token: str, source_cat: str,
                                    target_cat: str) -> Tuple[Optional[str], float]:
        src_vec = self.engine.get_token(source_token)
        if src_vec is None:
            return None, 0.0
        path = self.bundle.compute_geodesic(source_cat, target_cat, self.sheaf)
        transported = self.bundle.parallel_transport(src_vec, path)
        best_tok = None
        best_sim = -1.0
        for idx, (name, cat) in enumerate(zip(self.engine.token_names, self.engine.token_categories)):
            if cat == target_cat:
                sim = self.engine.sim(transported, self.engine.token_phases[idx])
                if sim > best_sim:
                    best_sim = sim
                    best_tok = name
        return best_tok, float(best_sim)
