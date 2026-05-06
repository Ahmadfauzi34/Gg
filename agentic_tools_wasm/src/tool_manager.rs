use wasm_bindgen::prelude::*;
use std::collections::{HashMap, HashSet};

const MAX_QUERY_TOKENS: usize = 16;
const TOP_K: usize = 8;

#[wasm_bindgen]
pub struct ToolManager {
    vocab: Vec<String>,
    vocab_map: HashMap<String, u16>,
    names: Vec<String>,
    names_lower: Vec<String>,
    tool_tokens: Vec<u16>,
    tool_starts: Vec<u32>,
    doc_lengths: Vec<u16>,
    df_table: Vec<u32>,
    idf_table: Vec<f32>,
    avgdl: f32,
    total_docs: u32,
    k1: f32,
    b: f32,
    threshold: f32,
    query_tokens_buf: Vec<u16>,
    scores_buf: Vec<f32>,
    topk_buf: [(f32, usize); TOP_K],
    temp_token_buf: Vec<u16>,
    freq_buf: [f32; MAX_QUERY_TOKENS],
    query_lower_buf: String,
}

#[wasm_bindgen]
impl ToolManager {
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        let mut manager = Self {
            vocab: Vec::new(),
            vocab_map: HashMap::new(),
            names: Vec::new(),
            names_lower: Vec::new(),
            tool_tokens: Vec::new(),
            tool_starts: Vec::new(),
            doc_lengths: Vec::new(),
            df_table: Vec::new(),
            idf_table: Vec::new(),
            avgdl: 0.0,
            total_docs: 0,
            k1: 1.2,
            b: 0.75,
            threshold: 0.1,
            query_tokens_buf: Vec::with_capacity(16),
            scores_buf: Vec::new(),
            topk_buf: [(-1.0, 0); TOP_K],
            temp_token_buf: Vec::with_capacity(64),
            freq_buf: [0.0; MAX_QUERY_TOKENS],
            query_lower_buf: String::with_capacity(128),
        };

        manager.add_tool("get_weather", &["cuaca", "suhu", "hujan", "ramalan", "weather", "forecast"]);
        manager.add_tool("get_time", &["waktu", "jam", "hari", "tanggal", "time", "clock"]);
        manager.add_tool("search_web", &["cari", "google", "browser", "internet", "search", "web"]);
        manager.add_tool("calculator", &["hitung", "kalkulator", "math", "tambah", "kurang", "calculator"]);
        manager.add_tool("text_stats", &["statistik", "teks", "kata", "karakter", "kalimat", "paragraf", "text", "stats", "count", "analyze"]);
        manager.add_tool("detect_language", &["deteksi", "bahasa", "language", "identify", "indo", "english", "detect"]);
        manager.add_tool("csv_to_json", &["csv", "json", "konversi", "tabel", "convert", "table", "parse"]);

        manager
    }

    #[wasm_bindgen]
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    #[wasm_bindgen]
    pub fn search_tools(&mut self, query: &str) -> js_sys::Array {
        let result = js_sys::Array::new();

        self.query_lower_buf.clear();
        self.query_lower_buf.extend(query.chars().flat_map(char::to_lowercase));

        let query_lower = self.query_lower_buf.clone();

        self.tokenize_query_from_lower(&query_lower);
        self.query_tokens_buf.truncate(MAX_QUERY_TOKENS);
        let q_len = self.query_tokens_buf.len();
        if q_len == 0 || self.total_docs == 0 {
            return result;
        }

        self.scores_buf.clear();
        self.scores_buf.resize(self.total_docs as usize, 0.0);
        self.topk_buf = [(-1.0, 0); TOP_K];

        let k1 = self.k1;
        let b = self.b;
        let avgdl = self.avgdl.max(1.0);
        let k1_plus_1 = k1 + 1.0;

        for tool_idx in 0..self.total_docs as usize {
            let mut score = 0.0_f32;
            let name_lower = &self.names_lower[tool_idx];

            if *name_lower == query_lower {
                score += 5.0;
            } else if name_lower.contains(&query_lower) {
                score += 2.5;
            }

            let start = self.tool_starts[tool_idx] as usize;
            let end = start + self.doc_lengths[tool_idx] as usize;
            let dl = f32::from(self.doc_lengths[tool_idx]);
            let norm = k1 * (1.0 - b + b * (dl / avgdl));

            self.freq_buf[..q_len].fill(0.0);

            count_token_freq_multi(
                &self.tool_tokens[start..end],
                &self.query_tokens_buf[..q_len],
                &mut self.freq_buf[..q_len],
            );

            for (j, &q_tok) in self.query_tokens_buf[..q_len].iter().enumerate() {
                let fqd = self.freq_buf[j];
                if fqd > 0.0 {
                    let idf = self.idf_table.get(q_tok as usize).copied().unwrap_or(0.0);
                    let tf_norm = (fqd * k1_plus_1) / (fqd + norm);
                    score += idf * tf_norm;
                }
            }

            self.scores_buf[tool_idx] = score;

            if score > self.threshold {
                self.insert_topk(score, tool_idx);
            }
        }

        for &(score, idx) in &self.topk_buf {
            if score < 0.0 { break; }
            result.push(&JsValue::from_str(&self.names[idx]));
        }

        result
    }

    #[inline]
    fn insert_topk(&mut self, score: f32, idx: usize) {
        let mut pos = TOP_K;
        for i in 0..TOP_K {
            if score > self.topk_buf[i].0 {
                pos = i;
                break;
            }
        }
        if pos < TOP_K {
            self.topk_buf.copy_within(pos..TOP_K - 1, pos + 1);
            self.topk_buf[pos] = (score, idx);
        }
    }

    fn add_tool(&mut self, name: &str, keywords: &[&str]) {
        let offset_start = self.tool_tokens.len() as u32;

        self.tokenize_to_ids(name);
        let name_ids = self.temp_token_buf.clone();
        for _ in 0..3 {
            self.tool_tokens.extend_from_slice(&name_ids);
        }

        for kw in keywords {
            self.tokenize_to_ids(kw);
            self.tool_tokens.extend_from_slice(&self.temp_token_buf);
        }

        let doc_len = (self.tool_tokens.len() as u32 - offset_start) as u16;
        self.tool_starts.push(offset_start);
        self.doc_lengths.push(doc_len);

        let mut seen = HashSet::new();
        for t in offset_start as usize..self.tool_tokens.len() {
            let tok_id = self.tool_tokens[t];
            if seen.insert(tok_id) {
                if (tok_id as usize) >= self.df_table.len() {
                    self.df_table.resize(tok_id as usize + 1, 0);
                }
                self.df_table[tok_id as usize] += 1;
            }
        }

        self.names.push(name.to_string());
        self.names_lower.push(name.to_lowercase());
        self.total_docs += 1;

        let total_len: u32 = self.doc_lengths.iter().map(|&d| u32::from(d)).sum();
        self.avgdl = total_len as f32 / self.total_docs as f32;

        self.rebuild_idf();
    }

    fn tokenize_query_from_lower(&mut self, lower: &str) {
        self.query_tokens_buf.clear();
        let bytes = lower.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            while i < bytes.len() && !bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_alphanumeric() {
                j += 1;
            }
            let word = unsafe { std::str::from_utf8_unchecked(&bytes[i..j]) };
            if let Some(&id) = self.vocab_map.get(word) {
                self.query_tokens_buf.push(id);
            }
            i = j;
        }
    }

    fn tokenize_to_ids(&mut self, text: &str) {
        self.temp_token_buf.clear();
        let lower = text.to_lowercase();
        let bytes = lower.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            while i < bytes.len() && !bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_alphanumeric() {
                j += 1;
            }
            let word = unsafe { std::str::from_utf8_unchecked(&bytes[i..j]) };

            let id = if let Some(&existing) = self.vocab_map.get(word) {
                existing
            } else {
                let new_id = self.vocab.len() as u16;
                let owned = word.to_string();
                self.vocab_map.insert(owned.clone(), new_id);
                self.vocab.push(owned);
                new_id
            };

            self.temp_token_buf.push(id);
            i = j;
        }
    }

    fn rebuild_idf(&mut self) {
        let n = self.total_docs as f32;
        let vocab_len = self.vocab.len();
        self.idf_table.resize(vocab_len, 0.0);
        self.df_table.resize(vocab_len, 0);

        for tok_id in 0..vocab_len {
            let nq = self.df_table[tok_id] as f32;
            let idf = ((n - nq + 0.5) / (nq + 0.5) + 1.0).ln();
            self.idf_table[tok_id] = idf.max(0.0);
        }
    }
}

impl Default for ToolManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// SIMD & Fallback Functions (Bebas Class)
// ============================================================

/// SIMD128-accelerated token frequency counting for BM25 sparse scoring.
/// Note: SIMD accelerates the O(n*m) frequency count phase, not the BM25 formula itself.
#[inline]
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn count_token_freq_multi(doc: &[u16], query: &[u16], out: &mut [f32]) {
    let q_len = query.len();
    for &t in doc {
        for j in 0..q_len {
            out[j] += f32::from(t == query[j]);
        }
    }
}

/// SIMD128-accelerated token frequency counting for BM25 sparse scoring.
/// Note: SIMD accelerates the O(n*m) frequency count phase, not the BM25 formula itself.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn count_token_freq_multi(doc: &[u16], query: &[u16], out: &mut [f32]) {
    use core::arch::wasm32::*;

    let len = doc.len();
    let q_len = query.len();
    debug_assert!(q_len <= MAX_QUERY_TOKENS);
    debug_assert!(q_len <= out.len());

    let mut i = 0;
    let mut accs: [v128; MAX_QUERY_TOKENS] = [i16x8_splat(0); MAX_QUERY_TOKENS];
    let mut q_splats: [v128; MAX_QUERY_TOKENS] = [i16x8_splat(0); MAX_QUERY_TOKENS];

    for j in 0..q_len {
        q_splats[j] = i16x8_splat(query[j] as i16);
    }

    while i + 8 <= len {
        let vec = unsafe { core::ptr::read_unaligned(doc.as_ptr().add(i) as *const v128) };
        for j in 0..q_len {
            let eq = i16x8_eq(vec, q_splats[j]);
            accs[j] = i16x8_sub(accs[j], eq);
        }
        i += 8;
    }

    for j in 0..q_len {
        let mut total = 0u32;
        total += i16x8_extract_lane::<0>(accs[j]) as u32;
        total += i16x8_extract_lane::<1>(accs[j]) as u32;
        total += i16x8_extract_lane::<2>(accs[j]) as u32;
        total += i16x8_extract_lane::<3>(accs[j]) as u32;
        total += i16x8_extract_lane::<4>(accs[j]) as u32;
        total += i16x8_extract_lane::<5>(accs[j]) as u32;
        total += i16x8_extract_lane::<6>(accs[j]) as u32;
        total += i16x8_extract_lane::<7>(accs[j]) as u32;
        out[j] += total as f32;
    }

    while i < len {
        let t = doc[i];
        for j in 0..q_len {
            out[j] += f32::from(t == query[j]);
        }
        i += 1;
    }
}
