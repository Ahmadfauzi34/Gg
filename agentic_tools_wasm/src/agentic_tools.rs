use wasm_bindgen::prelude::*;
use regex::Regex;
use base64::{Engine as _, engine::general_purpose};
use std::sync::OnceLock;

// ============================================================
// REGEX CACHE
// ============================================================

static URL_RE: OnceLock<Regex> = OnceLock::new();
static EMAIL_RE: OnceLock<Regex> = OnceLock::new();
static PHONE_RE: OnceLock<Regex> = OnceLock::new();
static CODE_BLOCK_RE: OnceLock<Regex> = OnceLock::new();
static VALID_EMAIL_RE: OnceLock<Regex> = OnceLock::new();
static SENTENCE_RE: OnceLock<Regex> = OnceLock::new();

fn get_or_init_regex<'a>(lock: &'a OnceLock<Regex>, pattern: &'static str) -> Result<&'a Regex, JsValue> {
    if let Some(re) = lock.get() {
        Ok(re)
    } else {
        let re = Regex::new(pattern)
            .map_err(|e| JsValue::from_str(&format!("Regex error: {e}")))?;
        let _ = lock.set(re);
        lock.get().ok_or_else(|| JsValue::from_str("Regex init race failed"))
    }
}

#[wasm_bindgen]
#[derive(Default)]
pub struct AgenticTools {}

#[wasm_bindgen]
impl AgenticTools {
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self { Self {} }

    /// Chunk text into overlapping segments (useful for RAG context windows)
    ///
    /// # Errors
    /// Returns an error if `chunk_size` is zero or overlap is >= `chunk_size`.
    #[wasm_bindgen]
    pub fn chunk_text(&self, text: &str, chunk_size: usize, overlap: usize) -> Result<js_sys::Array, JsValue> {
        let array = js_sys::Array::new();

        if chunk_size == 0 {
            return Err(JsValue::from_str("chunk_size cannot be zero"));
        }
        if overlap >= chunk_size {
            return Err(JsValue::from_str("overlap must be less than chunk_size"));
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(array);
        }

        let step = chunk_size - overlap;
        let mut i = 0;

        while i < words.len() {
            let end = std::cmp::min(i + chunk_size, words.len());
            let chunk = words[i..end].join(" ");
            array.push(&JsValue::from_str(&chunk));
            i += step;
        }

        Ok(array)
    }

    /// Truncate text using safe UTF-8 char counting instead of bytes
    #[wasm_bindgen]
    #[must_use]
    pub fn truncate_with_ellipsis(&self, text: &str, max_chars: usize) -> String {
        let char_count = text.chars().count();
        if char_count <= max_chars {
            return text.to_string();
        }
        if max_chars <= 3 {
            return text.chars().take(max_chars).collect();
        }

        let mut result = String::with_capacity(max_chars * 4); // worst-case UTF-8


        for (current_chars, c) in text.chars().enumerate() {
            if current_chars >= max_chars - 3 {
                break;
            }
            result.push(c);
        }
        result.push_str("...");
        result
    }

    /// Safely normalize whitespace while avoiding double allocations for preserve paragraphs mode.
    #[wasm_bindgen]
    #[must_use]
    pub fn normalize_whitespace(&self, text: &str, mode: &str) -> String {
        match mode {
            "preserve_paragraphs" => {
                let mut result = String::with_capacity(text.len());
                let mut first = true;

                for para in text.split("\n\n") {
                    let words: Vec<&str> = para.split_whitespace().collect();
                    if words.is_empty() { continue; }

                    if !first { result.push_str("\n\n"); }
                    result.push_str(&words.join(" "));
                    first = false;
                }
                result
            },
            _ => text.split_whitespace().collect::<Vec<&str>>().join(" "),
        }
    }

    /// Extract URLs from text
    ///
    /// # Errors
    /// Return an error if regex fails to compile.
    #[wasm_bindgen]
    pub fn extract_urls(&self, text: &str) -> Result<js_sys::Array, JsValue> {
        let re = get_or_init_regex(&URL_RE, r"https?://[^\s]+")?;
        let array = js_sys::Array::new();
        for cap in re.captures_iter(text) {
            array.push(&JsValue::from_str(&cap[0]));
        }
        Ok(array)
    }

    /// Analyze text sentiment (naive heuristic exact-match dictionary approach, no stemming)
    #[wasm_bindgen]
    #[must_use]
    pub fn analyze_sentiment(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();
        let positive_words = ["good", "great", "excellent", "awesome", "positive", "happy", "success"];
        let negative_words = ["bad", "terrible", "awful", "negative", "sad", "fail", "error"];

        let mut score = 0;
        for word in text_lower.split_whitespace() {
            if positive_words.contains(&word) {
                score += 1;
            } else if negative_words.contains(&word) {
                score -= 1;
            }
        }

        match score.cmp(&0) {
            std::cmp::Ordering::Greater => "positive".to_string(),
            std::cmp::Ordering::Less => "negative".to_string(),
            std::cmp::Ordering::Equal => "neutral".to_string(),
        }
    }

    #[wasm_bindgen]
    #[must_use]
    pub fn markdown_to_html(&self, markdown: &str) -> String {
        let parser = pulldown_cmark::Parser::new(markdown);
        let mut html_output = String::new();
        pulldown_cmark::html::push_html(&mut html_output, parser);
        html_output
    }

    #[wasm_bindgen]
    #[must_use]
    pub fn base64_encode(&self, text: &str) -> String {
        general_purpose::STANDARD.encode(text)
    }

    /// Decode Base64 to text
    ///
    /// # Errors
    /// Return an error if decoding fails.
    #[wasm_bindgen]
    pub fn base64_decode(&self, encoded: &str) -> Result<String, JsValue> {
        match general_purpose::STANDARD.decode(encoded) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(string) => Ok(string),
                Err(e) => Err(JsValue::from_str(&format!("Invalid UTF-8 sequence: {e}"))),
            },
            Err(e) => Err(JsValue::from_str(&format!("Invalid base64 string: {e}"))),
        }
    }

    /// Parse a JSON string and extract keys
    ///
    /// # Errors
    /// Return an error if json fails to parse.
    #[wasm_bindgen]
    pub fn extract_json_keys(&self, json_str: &str) -> Result<js_sys::Array, JsValue> {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(json_str);
        match parsed {
            Ok(serde_json::Value::Object(map)) => {
                let array = js_sys::Array::new();
                for key in map.keys() {
                    array.push(&JsValue::from_str(key));
                }
                Ok(array)
            },
            Ok(_) => Err(JsValue::from_str("JSON is not an object")),
            Err(e) => Err(JsValue::from_str(&format!("Invalid JSON: {e}"))),
        }
    }

    /// Calculate Cosine Similarity between two float arrays (for evaluating embeddings)
    ///
    /// # Errors
    /// Return an error if lengths do not match or empty.
    #[wasm_bindgen]
    pub fn cosine_similarity(&self, vec_a: &[f32], vec_b: &[f32]) -> Result<f32, JsValue> {
        if vec_a.len() != vec_b.len() {
            return Err(JsValue::from_str("Vectors must have the same length"));
        }
        if vec_a.is_empty() {
            return Err(JsValue::from_str("Vectors cannot be empty"));
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (a, b) in vec_a.iter().zip(vec_b.iter()) {
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        if norm_a == 0.0 || norm_b == 0.0 { return Ok(0.0); }
        Ok(dot_product / (norm_a.sqrt() * norm_b.sqrt()))
    }

    /// Mask Personal Identifiable Information (Emails and basic phone formats)
    ///
    /// # Errors
    /// Return an error if regex fails.
    #[wasm_bindgen]
    pub fn mask_pii(&self, text: &str) -> Result<String, JsValue> {
        let email_re = get_or_init_regex(&EMAIL_RE, r"([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")?;
        let mut masked = email_re.replace_all(text, "[EMAIL REDACTED]").to_string();

        let phone_re = get_or_init_regex(&PHONE_RE, r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")?;
        masked = phone_re.replace_all(&masked, "[PHONE REDACTED]").to_string();

        Ok(masked)
    }

    #[wasm_bindgen]
    #[must_use]
    pub fn strip_html(&self, html: &str) -> String {
        let mut result = String::with_capacity(html.len());
        let mut in_tag = false;
        for c in html.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(c),
                _ => {}
            }
        }
        result.trim().to_string()
    }

    /// Format a prompt template using variables from a JSON string
    /// Replace `{{key}}` with the corresponding value in the JSON object.
    ///
    /// # Errors
    /// Return an error if json fails to parse.
    #[wasm_bindgen]
    pub fn format_prompt(&self, template: &str, variables_json: &str) -> Result<String, JsValue> {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(variables_json);
        match parsed {
            Ok(serde_json::Value::Object(map)) => {
                let mut result = template.to_string();
                for (key, value) in &map {
                    let placeholder = format!("{{{{{key}}}}}");
                    let val_str = match value {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Number(n) => n.to_string(),
                        serde_json::Value::Bool(b) => b.to_string(),
                        _ => value.to_string(),
                    };
                    result = result.replace(&placeholder, &val_str);
                }
                Ok(result)
            },
            Ok(_) => Err(JsValue::from_str("Variables JSON must be an object")),
            Err(e) => Err(JsValue::from_str(&format!("Invalid JSON: {e}"))),
        }
    }

    #[wasm_bindgen]
    #[must_use]
    pub fn count_tokens_approx(&self, text: &str, method: &str) -> u32 {
        match method {
            "word" => text.split_whitespace().count() as u32,
            _ => ((text.len() as f32) / 4.0).ceil() as u32,
        }
    }

    /// Extract Code blocks
    ///
    /// # Errors
    /// Return an error if regex fails.
    #[wasm_bindgen]
    pub fn extract_code_blocks(&self, markdown: &str) -> Result<js_sys::Array, JsValue> {
        let re = get_or_init_regex(&CODE_BLOCK_RE, r"```(\w*)\n([\s\S]*?)```")?;
        let array = js_sys::Array::new();
        for cap in re.captures_iter(markdown) {
            let lang = cap.get(1).map_or("", |m: regex::Match| m.as_str());
            let code = cap.get(2).map_or("", |m: regex::Match| m.as_str());
            let entry = format!("{lang}|{code}");
            array.push(&JsValue::from_str(&entry));
        }
        Ok(array)
    }

    /// JSON Prettify
    ///
    /// # Errors
    /// Return an error if json parsing fails.
    #[wasm_bindgen]
    pub fn json_prettify(&self, json_str: &str) -> Result<String, JsValue> {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(json_str);
        match parsed {
            Ok(val) => match serde_json::to_string_pretty(&val) {
                Ok(s) => Ok(s),
                Err(e) => Err(JsValue::from_str(&format!("Serialization error: {e}"))),
            },
            Err(e) => Err(JsValue::from_str(&format!("Invalid JSON: {e}"))),
        }
    }

    /// JSON Minify
    ///
    /// # Errors
    /// Return an error if json parsing fails.
    #[wasm_bindgen]
    pub fn json_minify(&self, json_str: &str) -> Result<String, JsValue> {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(json_str);
        match parsed {
            Ok(val) => match serde_json::to_string(&val) {
                Ok(s) => Ok(s),
                Err(e) => Err(JsValue::from_str(&format!("Serialization error: {e}"))),
            },
            Err(e) => Err(JsValue::from_str(&format!("Invalid JSON: {e}"))),
        }
    }

    #[wasm_bindgen]
    #[must_use]
    pub fn levenshtein_distance(&self, s1: &str, s2: &str) -> u32 {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 { return len2 as u32; }
        if len2 == 0 { return len1 as u32; }

        let mut prev_row: Vec<usize> = (0..=len2).collect();
        let mut curr_row = vec![0usize; len2 + 1];

        for (i, c1) in s1.chars().enumerate() {
            curr_row[0] = i + 1;
            for (j, c2) in s2.chars().enumerate() {
                let cost = usize::from(c1 != c2);
                curr_row[j + 1] = (curr_row[j] + 1).min(prev_row[j + 1] + 1).min(prev_row[j] + cost);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }
        prev_row[len2] as u32
    }

    /// Extract Domain
    ///
    /// # Errors
    /// Return an error if extraction fails.
    #[wasm_bindgen]
    pub fn extract_domain(&self, url: &str) -> Result<String, JsValue> {
        let trimmed = url.trim();
        let without_proto = if let Some(pos) = trimmed.find("://") {
            &trimmed[pos + 3..]
        } else {
            trimmed
        };
        let domain = without_proto.split('/').next().unwrap_or("");
        let domain = domain.split(':').next().unwrap_or("");

        if domain.is_empty() {
            return Err(JsValue::from_str("Could not extract domain from URL"));
        }
        Ok(domain.to_lowercase())
    }

    #[wasm_bindgen]
    #[must_use]
    pub fn is_valid_email(&self, email: &str) -> bool {
        if let Ok(re) = get_or_init_regex(&VALID_EMAIL_RE, r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$") {
            re.is_match(email.trim())
        } else {
            false
        }
    }

    #[wasm_bindgen]
    #[must_use]
    pub fn slugify(&self, text: &str) -> String {
        let mut slug = String::with_capacity(text.len());
        let mut prev_dash = true;

        for c in text.to_lowercase().chars() {
            match c {
                'a'..='z' | '0'..='9' => {
                    slug.push(c);
                    prev_dash = false;
                },
                ' ' | '-' | '_' | '/' | '\\' | '.' | ',' | '!' | '?' | ':' | ';' => {
                    if !prev_dash {
                        slug.push('-');
                        prev_dash = true;
                    }
                },
                _ => {}
            }
        }
        if slug.ends_with('-') { slug.pop(); }
        slug
    }

    #[wasm_bindgen]
    #[must_use]
    pub fn estimate_reading_time(&self, text: &str) -> f32 {
        let word_count = text.split_whitespace().count() as f32;
        (word_count / 200.0).max(0.0)
    }

    /// Return text statistics as a JSON string.
    ///
    /// Fields: `words`, `chars`, `chars_no_spaces`, `sentences`, `paragraphs`.
    ///
    /// # Errors
    /// Returns an error if the sentence-split regex fails to compile.
    #[wasm_bindgen]
    pub fn text_stats(&self, text: &str) -> Result<String, JsValue> {
        let words = text.split_whitespace().count();
        let chars = text.chars().count();
        let chars_no_spaces = text.chars().filter(|c| !c.is_whitespace()).count();

        let sentence_re = get_or_init_regex(
            &SENTENCE_RE,
            r"[^.!?\n]+[.!?]+"
        )?;
        let sentences = sentence_re.find_iter(text).count().max(
            usize::from(!text.trim().is_empty())
        );

        let paragraphs = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .count();

        let json = format!(
            r#"{{"words":{words},"chars":{chars},"chars_no_spaces":{chars_no_spaces},"sentences":{sentences},"paragraphs":{paragraphs}}}"#
        );
        Ok(json)
    }

    /// Detect language using keyword heuristics.
    ///
    /// Returns `"id"` (Indonesian), `"en"` (English), or `"unknown"`.
    #[wasm_bindgen]
    #[must_use]
    pub fn detect_language(&self, text: &str) -> String {
        // Stopword lists — common, high-frequency words per language
        const ID_WORDS: &[&str] = &[
            "yang", "dan", "di", "ini", "itu", "dengan", "untuk", "tidak",
            "ada", "dari", "ke", "pada", "adalah", "juga", "sudah", "saya",
            "akan", "bisa", "atau", "kami", "kamu", "mereka", "anda",
        ];
        const EN_WORDS: &[&str] = &[
            "the", "and", "is", "in", "it", "of", "to", "that", "this",
            "was", "for", "on", "are", "with", "he", "she", "they", "we",
            "you", "have", "from", "not", "but", "what", "can", "been",
        ];

        let lower = text.to_lowercase();

        let mut id_score: u32 = 0;
        let mut en_score: u32 = 0;

        for word in lower.split_whitespace() {
            let w = word.trim_matches(|c: char| !c.is_alphabetic());
            if ID_WORDS.contains(&w) {
                id_score += 1;
            }
            if EN_WORDS.contains(&w) {
                en_score += 1;
            }
        }

        match id_score.cmp(&en_score) {
            std::cmp::Ordering::Greater => "id".to_string(),
            std::cmp::Ordering::Less    => "en".to_string(),
            std::cmp::Ordering::Equal   => {
                if id_score == 0 { "unknown".to_string() } else { "id".to_string() }
            }
        }
    }

    /// Parse a CSV string into a JSON array of objects.
    ///
    /// - First row is treated as the header.
    /// - Fields may be optionally quoted with `"`.
    /// - Empty input returns an empty JSON array `[]`.
    ///
    /// # Errors
    /// Returns an error if the resulting JSON cannot be serialised.
    #[wasm_bindgen]
    pub fn csv_to_json(&self, csv: &str) -> Result<String, JsValue> {
        let mut lines = csv.lines().filter(|l| !l.trim().is_empty());

        let Some(header_line) = lines.next() else { return Ok("[]".to_string()) };

        let headers: Vec<&str> = parse_csv_row(header_line);

        let mut rows: Vec<serde_json::Value> = Vec::new();

        for line in lines {
            let fields = parse_csv_row(line);
            let mut map = serde_json::Map::new();
            for (i, header) in headers.iter().enumerate() {
                let value = fields.get(i).copied().unwrap_or("");
                map.insert(
                    (*header).to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
            rows.push(serde_json::Value::Object(map));
        }

        serde_json::to_string(&rows)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
    }
}

// ============================================================
// CSV HELPERS (free function, not exposed to WASM)
// ============================================================

/// Parse one CSV row, respecting double-quoted fields.
fn parse_csv_row(line: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i <= len {
        if i == len {
            fields.push("");
            break;
        }
        if bytes[i] == b'"' {
            // Quoted field — find closing quote
            let start = i + 1;
            let mut j = start;
            while j < len {
                if bytes[j] == b'"' {
                    // Peek: double-quote escape?
                    if j + 1 < len && bytes[j + 1] == b'"' {
                        j += 2;
                    } else {
                        break;
                    }
                } else {
                    j += 1;
                }
            }
            // SAFETY: start..j are within original `line` bytes (valid UTF-8 slice)
            let field = core::str::from_utf8(&bytes[start..j]).unwrap_or("").trim();
            fields.push(field);
            // Skip past closing quote and comma
            i = j + 1;
            if i < len && bytes[i] == b',' {
                i += 1;
            }
        } else {
            // Unquoted field — scan to next comma
            let start = i;
            while i < len && bytes[i] != b',' {
                i += 1;
            }
            let field = core::str::from_utf8(&bytes[start..i]).unwrap_or("").trim();
            fields.push(field);
            i += 1; // skip comma
        }
    }

    fields
}
