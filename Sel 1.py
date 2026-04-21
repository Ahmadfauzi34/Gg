import numpy as np
from collections import defaultdict, deque
import time
from typing import Dict, List, Tuple, Optional, Any

class FHRREngine:
    def __init__(self, dim: int = 4096, n_hash_tables: int = 12, n_bins: int = 32):
        self.dim = dim
        self.n_tables = n_hash_tables
        self.n_bins = n_bins
        self.token_names: List[str] = []
        self.token_phases: List[np.ndarray] = []
        self.token_categories: List[str] = []
        self.token_polarities: List[int] = []
        self.role_names: List[str] = []
        self.role_phases: List[np.ndarray] = []
        self.poles: Dict[str, Dict[str, np.ndarray]] = {}
        self.pole_categories: set = set()
        self.transforms: Dict[str, Dict[str, Any]] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.rule_counter: int = 0
        self.lsh_tables: List[Dict[int, List[int]]] = [{} for _ in range(n_hash_tables)]
        self.lsh_projections: List[np.ndarray] = []
        for _ in range(n_hash_tables):
            proj = np.random.normal(0, 1, dim)
            proj = proj / np.linalg.norm(proj)
            self.lsh_projections.append(proj)
        self.episodic_capacity: int = 5000
        self.episodic_buffer: List[Dict[str, Any]] = []
        self.episodic_head: int = 0
        self._ws = np.zeros(dim)
        self._ws2 = np.zeros(dim)
        self._cbuf = np.zeros(dim, dtype=np.complex128)
        self.query_count = 0
        self.lsh_hits = 0
        self.linear_falls = 0
        # Training metrics
        self.learned_transforms_history: List[Dict] = []
        self.induced_rules_history: List[Dict] = []

    def alloc(self) -> np.ndarray:
        return np.random.uniform(-np.pi, np.pi, self.dim)

    def bind(self, a: np.ndarray, b: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = self._ws
        np.add(a, b, out=out)
        np.mod(out, 2 * np.pi, out=out)
        return out

    def unbind(self, bound: np.ndarray, probe: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = self._ws
        np.subtract(bound, probe, out=out)
        np.mod(out, 2 * np.pi, out=out)
        return out

    def bundle(self, vectors: List[np.ndarray], weights: Optional[np.ndarray] = None,
               out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = self._cbuf
        out.fill(0)
        if weights is None:
            weights = np.ones(len(vectors)) / len(vectors)
        for v, w in zip(vectors, weights):
            out += w * np.exp(1j * v)
        return np.angle(out)

    def sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.cos(a - b)))

    def _lsh_hash(self, vec: np.ndarray, table_idx: int) -> int:
        proj = self.lsh_projections[table_idx]
        scalar = np.dot(np.cos(vec), proj)
        bin_idx = int((scalar + 1.0) / 2.0 * self.n_bins)
        return max(0, min(bin_idx, self.n_bins - 1))

    def define_poles(self, category: str) -> Dict[str, np.ndarray]:
        if category in self.poles:
            return self.poles[category]
        pole_pos = self.alloc()
        pole_neg = (pole_pos + np.pi) % (2 * np.pi)
        self.poles[category] = {'positive': pole_pos, 'negative': pole_neg}
        self.pole_categories.add(category)
        return self.poles[category]

    def add_token(self, name: str, category: str, polarity: int = 0,
                  prototype: Optional[np.ndarray] = None, noise: float = 0.20) -> np.ndarray:
        if name in self.token_names:
            return self.token_phases[self.token_names.index(name)]
        if prototype is None:
            prototype = self.alloc()
        vec = (prototype + np.random.normal(0, noise, self.dim)) % (2 * np.pi)
        if polarity != 0 and category in self.poles:
            pole_vec = self.poles[category]['positive'] if polarity > 0 else self.poles[category]['negative']
            vec = (vec + pole_vec * 0.4) % (2 * np.pi)
        idx = len(self.token_names)
        self.token_names.append(name)
        self.token_phases.append(vec)
        self.token_categories.append(category)
        self.token_polarities.append(polarity)
        for t in range(self.n_tables):
            bin_idx = self._lsh_hash(vec, t)
            if bin_idx not in self.lsh_tables[t]:
                self.lsh_tables[t][bin_idx] = []
            self.lsh_tables[t][bin_idx].append(idx)
        return vec

    def get_token(self, name: str) -> Optional[np.ndarray]:
        try:
            idx = self.token_names.index(name)
            return self.token_phases[idx]
        except ValueError:
            return None

    def add_role(self, name: str) -> np.ndarray:
        if name in self.role_names:
            return self.role_phases[self.role_names.index(name)]
        vec = self.alloc()
        self.role_names.append(name)
        self.role_phases.append(vec)
        return vec

    def get_role(self, name: str) -> Optional[np.ndarray]:
        try:
            idx = self.role_names.index(name)
            return self.role_phases[idx]
        except ValueError:
            return None

    def encode(self, bindings: Dict[str, str]) -> Optional[np.ndarray]:
        bound_vecs = []
        for role_name, token_name in bindings.items():
            role_vec = self.get_role(role_name)
            token_vec = self.get_token(token_name)
            if role_vec is None or token_vec is None:
                continue
            bound = self.bind(role_vec, token_vec, out=self._ws2.copy())
            bound_vecs.append(bound)
        if not bound_vecs:
            return None
        return self.bundle(bound_vecs)

    def decode(self, struct_vec: np.ndarray, threshold: float = 0.40) -> Dict[str, Tuple[str, float]]:
        decomposition = {}
        for role_name, role_vec in zip(self.role_names, self.role_phases):
            unbound = self.unbind(struct_vec, role_vec, out=self._ws.copy())
            best_idx = -1
            best_sim = -1.0
            for idx, vec in enumerate(self.token_phases):
                sim = self.sim(unbound, vec)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            if best_sim > threshold:
                decomposition[role_name] = (self.token_names[best_idx], best_sim)
        return decomposition

    def cleanup(self, query_vec: np.ndarray, threshold: float = 0.45,
                probe_factor: int = 2) -> Tuple[Optional[str], float]:
        self.query_count += 1
        candidates = set()
        for t in range(self.n_tables):
            bin_idx = self._lsh_hash(query_vec, t)
            if bin_idx in self.lsh_tables[t]:
                candidates.update(self.lsh_tables[t][bin_idx])
        if len(candidates) < probe_factor * 3:
            for t in range(self.n_tables):
                bin_idx = self._lsh_hash(query_vec, t)
                for delta in [-1, 1]:
                    neighbor = bin_idx + delta
                    if 0 <= neighbor < self.n_bins and neighbor in self.lsh_tables[t]:
                        candidates.update(self.lsh_tables[t][neighbor])
        if len(candidates) < 5:
            self.linear_falls += 1
            candidates = range(len(self.token_names))
        else:
            self.lsh_hits += 1
        best_name = None
        best_sim = -1.0
        for idx in candidates:
            sim = self.sim(query_vec, self.token_phases[idx])
            if sim > best_sim:
                best_sim = sim
                best_name = self.token_names[idx]
        if best_sim > threshold:
            return best_name, best_sim
        return None, best_sim

    def kernel_query(self, query_vec: np.ndarray, radius: float = 0.3,
                     top_k: int = 5) -> List[Tuple[str, float, float]]:
        results = []
        for idx, vec in enumerate(self.token_phases):
            sim = self.sim(query_vec, vec)
            weight = np.exp(-((1.0 - sim) ** 2) / (2 * radius ** 2))
            if weight > 0.1:
                results.append((self.token_names[idx], sim, weight))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def store_episodic(self, vec: np.ndarray, metadata: Optional[Dict] = None):
        entry = {
            'vector': vec.copy(),
            'timestamp': time.time(),
            'access_count': 0,
            'decay_factor': 1.0,
            'metadata': metadata or {}
        }
        if len(self.episodic_buffer) >= self.episodic_capacity:
            self.episodic_buffer[self.episodic_head] = entry
            self.episodic_head = (self.episodic_head + 1) % self.episodic_capacity
        else:
            self.episodic_buffer.append(entry)

    def apply_decay(self, current_time: Optional[float] = None, decay_rate: float = 0.001):
        if current_time is None:
            current_time = time.time()
        for entry in self.episodic_buffer:
            age = current_time - entry['timestamp']
            retention = np.exp(-decay_rate * age) * (1 + 0.1 * entry['access_count'])
            entry['decay_factor'] = retention
            if retention < 0.9:
                noise_level = (1.0 - retention) * 0.1
                entry['vector'] = (entry['vector'] +
                                  np.random.normal(0, noise_level, self.dim)) % (2 * np.pi)

    def query_episodic(self, query_vec: np.ndarray, threshold: float = 0.4):
        best_match = None
        best_score = -1.0
        for entry in self.episodic_buffer:
            raw_sim = self.sim(query_vec, entry['vector'])
            compensated = raw_sim * entry['decay_factor']
            if compensated > best_score:
                best_score = compensated
                best_match = entry
        if best_match and best_score > threshold:
            best_match['access_count'] += 1
            return best_match, best_score
        return None, best_score

    def learn_transform(self, name: str, source_vec: np.ndarray,
                        target_vec: np.ndarray, min_confidence: float = 0.5) -> Tuple[Optional[str], float]:
        diff = (target_vec - source_vec + np.pi) % (2 * np.pi) - np.pi
        test = (source_vec + diff) % (2 * np.pi)
        conf = self.sim(test, target_vec)
        if conf < min_confidence:
            return None, conf
        self.transforms[name] = {'vector': diff, 'confidence': conf, 'usage_count': 0,
                                 'source': None, 'target': None}  # akan diisi oleh trainer
        return name, conf

    def apply_transform(self, source_vec: np.ndarray, transform_name: str,
                        scale: float = 1.0) -> Optional[np.ndarray]:
        if transform_name not in self.transforms:
            return None
        t = self.transforms[transform_name]
        result = (source_vec + t['vector'] * scale) % (2 * np.pi)
        t['usage_count'] += 1
        return result

    def chain_transforms(self, source_vec: np.ndarray,
                         transform_names: List[str]) -> Tuple[Optional[np.ndarray], float]:
        current = source_vec.copy()
        total_conf = 1.0
        for tname in transform_names:
            if tname not in self.transforms:
                return None, 0.0
            t = self.transforms[tname]
            current = (current + t['vector']) % (2 * np.pi)
            total_conf *= t['confidence']
        return current, total_conf

    def add_rule(self, pattern_bindings: Dict[str, str], action: str,
                 transform_name: Optional[str] = None, confidence: float = 0.7,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        pattern_vec = self.encode(pattern_bindings)
        if pattern_vec is None:
            raise ValueError("Invalid pattern bindings")
        rid = f"rule_{self.rule_counter}"
        self.rule_counter += 1
        self.rules[rid] = {
            'pattern': pattern_vec, 'action': action, 'confidence': confidence,
            'support': 1, 'bindings': pattern_bindings, 'transform': transform_name,
            'metadata': metadata or {}
        }
        self.induced_rules_history.append({
            'id': rid, 'bindings': pattern_bindings, 'action': action,
            'confidence': confidence, 'timestamp': time.time(), 'metadata': metadata or {}
        })
        return rid

    def match_rule(self, query_vec: np.ndarray, threshold: float = 0.65,
                   metadata_filter: Optional[Dict[str, Any]] = None):
        best_rid = None
        best_sim = -1.0
        for rid, rule in self.rules.items():
            # Apply metadata filter if provided
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    if rule['metadata'].get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            sim = self.sim(query_vec, rule['pattern'])
            if sim > best_sim:
                best_sim = sim
                best_rid = rid
        if best_rid and best_sim > threshold:
            return self.rules[best_rid], best_sim
        return None, best_sim

    def detect_contradiction(self, vec1: np.ndarray, vec2: np.ndarray,
                             threshold: float = 0.0) -> Tuple[bool, List[Dict]]:
        decomp1 = self.decode(vec1, threshold=0.35)
        decomp2 = self.decode(vec2, threshold=0.35)
        conflicts = []
        for role in set(decomp1.keys()) & set(decomp2.keys()):
            f1, c1 = decomp1[role]
            f2, c2 = decomp2[role]
            sim = self.sim(self.get_token(f1), self.get_token(f2))
            if sim < threshold:
                conflicts.append({
                    'role': role, 'filler1': f1, 'filler2': f2,
                    'filler_sim': sim, 'severity': abs(sim)
                })
        return len(conflicts) > 0, conflicts

    # -------------------------------------------------------------------------
    # DATASET INTEGRATION (v2)
    # -------------------------------------------------------------------------

    def build_from_dataset(self, dataset: Dict[str, Any]):
        vocab = dataset.get('vocab', {})
        cat_map = vocab.get('categories', {})
        pole_map = vocab.get('poles', {})

        for pole_cat in pole_map.keys():
            self.define_poles(pole_cat)

        prototypes = {cat: self.alloc() for cat in cat_map.keys()}

        for cat_name, tokens in cat_map.items():
            proto = prototypes[cat_name]
            pole_cat = cat_name if cat_name in pole_map else None
            for tok in tokens:
                polarity = 0
                if pole_cat:
                    if tok in pole_map[pole_cat].get('positive', []):
                        polarity = +1
                    elif tok in pole_map[pole_cat].get('negative', []):
                        polarity = -1
                self.add_token(tok, cat_name, polarity, proto,
                               noise=0.18 if polarity != 0 else 0.22)

        # Roles dari dataset (auto-detect dari observations + qa_pairs)
        all_roles = set()
        for obs in dataset.get('observations', []):
            all_roles.update(obs.get('bindings', {}).keys())
        for qa in dataset.get('qa_pairs', []):
            all_roles.add(qa.get('answer_role'))
        for r in sorted(all_roles):
            if r:
                self.add_role(r)

        print(f"[Engine] Vocab: {len(self.token_names)} tokens, {len(self.role_names)} roles")

    def learn_transform_from_data(self, name: str, from_token: str, to_token: str,
                                  min_confidence: float = 0.5) -> Tuple[Optional[str], float]:
        src = self.get_token(from_token)
        tgt = self.get_token(to_token)
        if src is None or tgt is None:
            return None, 0.0
        result, conf = self.learn_transform(name, src, tgt, min_confidence)
        if result:
            self.transforms[name]['source'] = from_token
            self.transforms[name]['target'] = to_token
            self.learned_transforms_history.append({
                'name': name, 'from': from_token, 'to': to_token,
                'confidence': conf, 'timestamp': time.time()
            })
        return result, conf
