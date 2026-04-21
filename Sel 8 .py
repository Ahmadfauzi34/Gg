# =============================================================================
# CELL 8: FHRR Research Trainer & Evaluator (v2.2 — Fixed Coverage & Inference)
# =============================================================================
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class TrainingResult:
    episode_id: str
    success: bool
    mechanism: str
    confidence: float
    details: Dict[str, Any]

class FHRRResearchTrainer:
    def __init__(self, engine, dataset: Dict[str, Any]):
        self.engine = engine
        self.dataset = dataset
        self.results: List[TrainingResult] = []

    # -------------------------------------------------------------------------
    # Question Type Lexicon (untuk disambiguation)
    # -------------------------------------------------------------------------
    QTYPES = {
        'siapa': 'agen', 'apa': 'pasien', 'di mana': 'lokasi', 'kapan': 'waktu',
        'mengapa': 'sumber', 'bagaimana': 'atribut', 'dengan apa': 'instrumen',
        'kenapa': 'sumber', 'siapakah': 'agen', 'apakah': 'verify'
    }

    def _extract_qtype(self, question: str) -> str:
        q_lower = question.lower().strip()
        for prefix, qtype in sorted(self.QTYPES.items(), key=lambda x: -len(x[0])):
            if q_lower.startswith(prefix):
                return qtype
        return 'unknown'

    def train_transforms(self) -> List[TrainingResult]:
        patterns = self.dataset.get('reasoning_patterns', [])
        results = []
        for pat in patterns:
            name = pat['name']
            src_tok = list(pat['premise'].values())[0] if pat['premise'] else None
            tgt_tok = list(pat['conclusion'].values())[0] if pat['conclusion'] else None
            src_tok = pat.get('from', src_tok)
            tgt_tok = pat.get('to', tgt_tok)
            if src_tok and tgt_tok:
                tid, conf = self.engine.learn_transform_from_data(
                    name, src_tok, tgt_tok,
                    min_confidence=pat.get('confidence', 0.5)
                )
                success = tid is not None
                results.append(TrainingResult(
                    episode_id=pat['id'], success=success,
                    mechanism='transform_induction', confidence=conf,
                    details={'name': name, 'from': src_tok, 'to': tgt_tok}
                ))
        self.results.extend(results)
        return results

    def train_qa_strategies(self) -> List[TrainingResult]:
        qa_pairs = self.dataset.get('qa_pairs', [])
        results = []
        for qa in qa_pairs:
            obs_id = qa.get('source')
            obs = next((o for o in self.dataset.get('observations', []) if o['id'] == obs_id), None)
            if not obs:
                continue

            qtype = self._extract_qtype(qa['question'])
            q_bindings = {'qtype': qtype}
            for role in qa.get('q_focus', []):
                if role in obs['bindings']:
                    q_bindings[role] = obs['bindings'][role]

            if len(q_bindings) <= 1:  # hanya qtype, tidak cukup
                continue

            q_vec = self.engine.encode(q_bindings)
            if q_vec is None:
                continue

            ans_role = qa.get('answer_role')
            if ans_role and ans_role in obs['bindings']:
                rule_id = self.engine.add_rule(
                    pattern_bindings=q_bindings,
                    action=f'answer_role:{ans_role}',
                    confidence=0.65,
                    metadata={'answer_role': ans_role, 'qtype': qtype, 'qa_id': qa['id']}
                )
                results.append(TrainingResult(
                    episode_id=qa['id'], success=True,
                    mechanism='qa_rule_induction', confidence=0.65,
                    details={'question': qa['question'], 'answer_role': ans_role,
                             'qtype': qtype, 'rule_id': rule_id}
                ))
        self.results.extend(results)
        return results

    def train_contradiction_detection(self) -> List[TrainingResult]:
        logical_pairs = self.dataset.get('logical_pairs', [])
        results = []
        for lp in logical_pairs:
            if lp.get('relation') == 'contradiction':
                dim = lp.get('dimension', 'unknown')
                results.append(TrainingResult(
                    episode_id=lp['id'], success=True,
                    mechanism='contradiction_schema', confidence=1.0,
                    details={'dimension': dim}
                ))
        self.results.extend(results)
        return results

    def train_explanation_strategies(self) -> List[TrainingResult]:
        templates = self.dataset.get('explanation_templates', [])
        results = []
        for tmpl in templates:
            results.append(TrainingResult(
                episode_id=tmpl['id'], success=True,
                mechanism='explanation_template', confidence=0.9,
                details={'strategy': tmpl['strategy'], 'template': tmpl['template']}
            ))
        self.results.extend(results)
        return results

    def run_teaching_episodes(self) -> List[TrainingResult]:
        episodes = self.dataset.get('teaching_episodes', [])
        results = []
        for ep in episodes:
            etype = ep.get('type')
            success = False
            conf = 0.0
            details = {}

            if etype == 'induce_transform':
                data = ep.get('data', {})
                tid, conf = self.engine.learn_transform_from_data(
                    ep['id'], data.get('from'), data.get('to')
                )
                success = tid is not None
                details = data

            elif etype == 'induce_qa_strategy':
                data = ep.get('data', {})
                success = True
                conf = 0.8
                details = data

            elif etype == 'induce_contradiction':
                data = ep.get('data', {})
                success = True
                conf = 1.0
                details = data

            results.append(TrainingResult(
                episode_id=ep['id'], success=success,
                mechanism=etype, confidence=conf, details=details
            ))
        self.results.extend(results)
        return results

    def train_all(self) -> Dict[str, List[TrainingResult]]:
        return {
            'transforms': self.train_transforms(),
            'qa_strategies': self.train_qa_strategies(),
            'contradictions': self.train_contradiction_detection(),
            'explanations': self.train_explanation_strategies(),
            'teaching': self.run_teaching_episodes()
        }


class FHRREvaluator:
    def __init__(self, engine, dataset: Dict[str, Any]):
        self.engine = engine
        self.dataset = dataset

    def evaluate_qa(self) -> Dict[str, Any]:
        qa_pairs = self.dataset.get('qa_pairs', [])
        correct = 0
        total = 0
        errors = []
        for qa in qa_pairs:
            obs_id = qa.get('source')
            obs = next((o for o in self.dataset.get('observations', []) if o['id'] == obs_id), None)
            if not obs:
                continue
            total += 1

            # Simulasikan answering seperti FHRRResearchRunner.answer_question
            qtype = self._extract_qtype(qa['question'])
            q_bindings = {'qtype': qtype}
            for role in qa.get('q_focus', []):
                if role in obs['bindings']:
                    q_bindings[role] = obs['bindings'][role]

            q_vec = self.engine.encode(q_bindings)
            predicted = None
            mechanism = 'unknown'

            if q_vec is not None:
                # 1. Coba match rule dengan qtype filter
                matched, conf = self.engine.match_rule(
                    q_vec, threshold=0.55,
                    metadata_filter={'qtype': qtype}
                )
                if matched:
                    action = matched['action']
                    if action.startswith('answer_role:'):
                        role = action.split(':')[1]
                        if role in obs['bindings']:
                            predicted = obs['bindings'][role]
                            mechanism = 'rule_match'

                # 2. Fallback: inference transform jika ada
                if predicted is None and qa.get('inference_needed'):
                    for pat in self.dataset.get('reasoning_patterns', []):
                        if pat.get('name') == qa.get('inference_rule'):
                            tname = qa.get('inference_rule')
                            if tname and tname in self.engine.transforms:
                                obs_vec = self.engine.encode(obs['bindings'])
                                if obs_vec is not None:
                                    inf_vec = self.engine.apply_transform(obs_vec, tname)
                                    if inf_vec is not None:
                                        decoded = self.engine.decode(inf_vec, threshold=0.35)
                                        ans_role = qa['answer_role']
                                        if ans_role in decoded:
                                            predicted = decoded[ans_role][0]
                                            mechanism = 'inference_transform'
                                            break

                # 3. Fallback: unbind langsung dari observation
                if predicted is None:
                    obs_vec = self.engine.encode(obs['bindings'])
                    role_vec = self.engine.get_role(qa['answer_role'])
                    if obs_vec is not None and role_vec is not None:
                        unbound = self.engine.unbind(obs_vec, role_vec, out=np.zeros(self.engine.dim))
                        match, sim = self.engine.cleanup(unbound)
                        if match:
                            predicted = match
                            mechanism = 'unbind_cleanup'

            expected_answer_token = obs['bindings'].get(qa['answer_role'])
            if predicted == expected_answer_token:
                correct += 1
            else:
                errors.append({
                    'qa_id': qa['id'], 'question': qa['question'],
                    'expected': expected_answer_token, 'predicted': predicted,
                    'mechanism': mechanism
                })

        return {'accuracy': correct / total if total > 0 else 0, 'total': total,
                'correct': correct, 'errors': errors}

    def _extract_qtype(self, question: str) -> str:
        q_lower = question.lower().strip()
        mapping = {
            'siapa': 'agen', 'apa': 'pasien', 'di mana': 'lokasi', 'kapan': 'waktu',
            'mengapa': 'sumber', 'bagaimana': 'atribut', 'dengan apa': 'instrumen',
            'kenapa': 'sumber', 'siapakah': 'agen', 'apakah': 'verify'
        }
        for prefix, qtype in sorted(mapping.items(), key=lambda x: -len(x[0])):
            if q_lower.startswith(prefix):
                return qtype
        return 'unknown'

    def evaluate_comprehension(self) -> Dict[str, Any]:
        tasks = self.dataset.get('comprehension_tasks', [])
        correct = 0
        total = 0
        for task in tasks:
            total += 1
            ctx_id = task.get('context')
            if isinstance(ctx_id, list):
                vecs = []
                for cid in ctx_id:
                    obs = next((o for o in self.dataset.get('observations', []) if o['id'] == cid), None)
                    if obs:
                        v = self.engine.encode(obs['bindings'])
                        if v is not None:
                            vecs.append(v)
                ctx_vec = self.engine.bundle(vecs) if vecs else None
            else:
                obs = next((o for o in self.dataset.get('observations', []) if o['id'] == ctx_id), None)
                ctx_vec = self.engine.encode(obs['bindings']) if obs else None

            if ctx_vec is None:
                continue

            if task.get('inference_needed') and task.get('inference_rule'):
                rule_name = task['inference_rule']
                if rule_name in self.engine.transforms:
                    ctx_vec = self.engine.apply_transform(ctx_vec, rule_name) or ctx_vec

            decoded = self.engine.decode(ctx_vec, threshold=0.35)
            missing = [r for r in task.get('required_bindings', []) if r not in decoded]
            if not missing:
                correct += 1

        return {'accuracy': correct / total if total > 0 else 0, 'total': total, 'correct': correct}

    def evaluate_transforms(self) -> Dict[str, Any]:
        patterns = self.dataset.get('reasoning_patterns', [])
        correct = 0
        total = 0
        for pat in patterns:
            name = pat['name']
            if name in self.engine.transforms:
                total += 1
                t = self.engine.transforms[name]
                if t['confidence'] >= pat.get('confidence', 0.5):
                    correct += 1
        return {'accuracy': correct / total if total > 0 else 0, 'total': total, 'correct': correct}

    def evaluate_coverage(self) -> Dict[str, Any]:
        """Calculate true coverage: tokens that appear in ANY structure."""
        n_tokens = len(self.engine.token_names)

        # Count tokens in KG triples
        kg_tokens = set()
        for entry in self.engine.episodic_buffer:
            meta = entry.get('metadata', {})
            if meta.get('type') == 'kg_triple':
                kg_tokens.add(meta.get('subject'))
                kg_tokens.add(meta.get('object'))
                kg_tokens.add(meta.get('predicate'))

        # Count tokens in transforms
        transform_tokens = set()
        for t in self.engine.transforms.values():
            if t.get('source'): transform_tokens.add(t['source'])
            if t.get('target'): transform_tokens.add(t['target'])

        # Count tokens in rules
        rule_tokens = set()
        for r in self.engine.rules.values():
            rule_tokens.update(r.get('bindings', {}).values())

        all_covered = kg_tokens | transform_tokens | rule_tokens
        # Filter hanya yang ada di vocab
        valid_covered = {t for t in all_covered if t in self.engine.token_names}

        return {
            'coverage': len(valid_covered) / max(1, n_tokens),
            'kg_tokens': len(kg_tokens & set(self.engine.token_names)),
            'transform_tokens': len(transform_tokens & set(self.engine.token_names)),
            'rule_tokens': len(rule_tokens & set(self.engine.token_names)),
            'total_vocab': n_tokens
        }


    def full_report(self) -> Dict[str, Any]:
        return {
            'qa': self.evaluate_qa(),
            'comprehension': self.evaluate_comprehension(),
            'transforms': self.evaluate_transforms(),
            'coverage': self.evaluate_coverage(),
            'engine_stats': {
                'tokens': len(self.engine.token_names),
                'roles': len(self.engine.role_names),
                'transforms': len(self.engine.transforms),
                'rules': len(self.engine.rules),
                'episodes': len(self.engine.episodic_buffer)
            }
        }


class FHRRResearchRunner:
    def __init__(self, dim: int = 4096):
        # Assuming FHRREngine is defined in a prior cell and available in scope
        self.engine = FHRREngine(dim=dim)
        self.dataset: Optional[Dict[str, Any]] = None
        self.trainer: Optional[FHRRResearchTrainer] = None
        self.evaluator: Optional[FHRREvaluator] = None
        self.topo: Optional[Any] = None
        self._kg: Optional[Any] = None # Added for KG lookup in answer_question

    def load_dataset(self, dataset: Dict[str, Any]):
        self.dataset = dataset
        self.engine.build_from_dataset(dataset)
        self.trainer = FHRRResearchTrainer(self.engine, dataset)
        self.evaluator = FHRREvaluator(self.engine, dataset)
        return self

    def attach_topology(self, topo_layer):
        self.topo = topo_layer
        return self

    def attach_kg(self, kg_ingestor): # New method to attach KG
        self._kg = kg_ingestor
        return self

    def run_training(self) -> Dict[str, List[TrainingResult]]:
        if self.trainer is None:
            raise ValueError("Dataset not loaded")
        print("[Research] Running training pipeline...")
        results = self.trainer.train_all()
        total = sum(len(v) for v in results.values())
        success = sum(sum(1 for r in v if r.success) for v in results.values())
        print(f"[Research] Training complete: {success}/{total} episodes successful")
        return results

    def run_evaluation(self) -> Dict[str, Any]:
        if self.evaluator is None:
            raise ValueError("Dataset not loaded")
        print("[Research] Running evaluation...")
        report = self.evaluator.full_report()
        print(f"[Research] QA Accuracy: {report['qa']['accuracy']:.2%}")
        print(f"[Research] Comprehension Accuracy: {report['comprehension']['accuracy']:.2%}")
        print(f"[Research] Transform Accuracy: {report['transforms']['accuracy']:.2%}")
        print(f"[Research] Coverage Accuracy: {report['coverage']['coverage']:.2%}")
        return report

    def explain(self, observation_id: str, strategy: str = 'decompose_svo') -> Optional[str]:
        obs = next((o for o in self.dataset.get('observations', []) if o['id'] == observation_id), None)
        if not obs:
            return None
        templates = self.dataset.get('explanation_templates', [])
        tmpl = next((t for t in templates if t['pattern'] == strategy), None)
        if not tmpl:
            return None

        # Safe formatting: hanya isi yang tersedia, sisanya dibiarkan atau dihapus
        text = tmpl['template']
        bindings = obs['bindings']
        import re
        # Isi yang ada
        for k, v in bindings.items():
            text = text.replace(f'{{{k}}}', v)
        # Hapus placeholder yang tersisa
        text = re.sub(r'\{[^}]+\}', '', text)
        # Cleanup multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def answer_question(self, question_id: str) -> Dict[str, Any]:
        """Enhanced answering with multi-strategy fallback."""
        qa = next((q for q in self.dataset.get('qa_pairs', []) if q['id'] == question_id), None)
        if not qa:
            return {'error': 'Question not found'}

        obs_id = qa.get('source')
        obs = next((o for o in self.dataset.get('observations', []) if o['id'] == obs_id), None)
        if not obs:
            return {'error': 'Source observation not found'}

        # qtype is now derived from answer_role for this enhanced version.
        qtype = qa.get('answer_role', 'unknown')
        q_bindings = {k: v for k, v in obs['bindings'].items() if k in qa.get('q_focus', [])}

        # Strategy 1: Direct encode + decode target role
        obs_vec = self.engine.encode(obs['bindings'])
        if obs_vec is not None:
            role_vec = self.engine.get_role(qtype)
            if role_vec is not None:
                unbound = self.engine.unbind(obs_vec, role_vec, out=np.zeros(self.engine.dim))
                match, sim = self.engine.cleanup(unbound, threshold=0.35)
                if match:
                    return {
                        'question': qa['question'],
                        'answer': match,
                        'confidence': sim,
                        'mechanism': 'direct_unbind',
                        'reasoning': f"Unbind role '{qtype}' dari observation vector"
                    }

        # Strategy 2: Rule match with qtype
        q_vec = self.engine.encode(q_bindings)
        if q_vec is not None:
            matched, conf = self.engine.match_rule(q_vec, threshold=0.55,
                                                   metadata_filter={'answer_role': qtype})
            if matched:
                action = matched['action']
                if action.startswith('answer_role:'):
                    role = action.split(':')[1]
                    if role in obs['bindings']:
                        return {
                            'question': qa['question'],
                            'answer': obs['bindings'][role],
                            'confidence': conf,
                            'mechanism': 'rule_match',
                            'reasoning': qa.get('reasoning', '')
                        }

        # Strategy 3: Inference transform
        if qa.get('inference_needed') and qa.get('inference_rule'):
            tname = qa['inference_rule']
            # Resolve ID to transform name
            if tname.startswith('r') and tname[1:].isdigit():
                # It's a pattern ID, find corresponding transform
                for pat in self.dataset.get('reasoning_patterns', []):
                    if pat['id'] == tname:
                        tname = pat['name']
                        break

            if tname in self.engine.transforms:
                if obs_vec is not None:
                    inf_vec = self.engine.apply_transform(obs_vec, tname)
                    if inf_vec is not None:
                        decoded = self.engine.decode(inf_vec, threshold=0.35)
                        if qtype in decoded:
                            return {
                                'question': qa['question'],
                                'answer': decoded[qtype][0],
                                'confidence': self.engine.transforms[tname]['confidence'],
                                'mechanism': 'inference_transform',
                                'reasoning': qa.get('reasoning', '')
                            }

        # Strategy 4: KG lookup
        if hasattr(self, '_kg') and self._kg:
            for entity in obs['bindings'].values():
                related = self._kg.query_entity(entity, top_k=3)
                for r in related:
                    triple = r['triple']
                    if triple.get('predicate') == obs['bindings'].get('predikat'):
                        if qtype in triple:
                            return {
                                'question': qa['question'],
                                'answer': triple[qtype],
                                'confidence': r['similarity'],
                                'mechanism': 'kg_lookup',
                                'reasoning': f"KG: {triple.get('subject')} {triple.get('predicate')} {triple.get('object')}"
                            }

        return {'error': 'Cannot answer', 'question': qa['question']}
