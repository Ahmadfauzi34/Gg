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

macro_rules! get_or_init_regex {
    ($lock:ident, $pattern:expr) => {
        {
            let res: Result<&Regex, JsValue> = if let Some(re) = $lock.get() {
                Ok(re)
            } else {
                match init_regex($pattern) {
                    Ok(re) => {
                        let _ = $lock.set(re);
                        // Safe to unwrap because we just set it
                        #[allow(clippy::unwrap_used)]
                        let ret = $lock.get().unwrap();
                        Ok(ret)
                    },
                    Err(e) => Err(e),
                }
            };
            res
        }
    }
}

fn init_regex(pattern: &str) -> Result<Regex, JsValue> {
    Regex::new(pattern).map_err(|e| JsValue::from_str(&format!("Regex error: {e}")))
}

#[wasm_bindgen]
#[derive(Default)]
pub struct AgenticTools {}

#[wasm_bindgen]
impl AgenticTools {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AgenticTools {
        AgenticTools {}
    }

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if regex fails to compile.
    pub fn extract_urls(&self, text: &str) -> Result<js_sys::Array, JsValue> {
        let re = get_or_init_regex!(URL_RE, r"https?://[^\s]+")?;
        let array = js_sys::Array::new();
        for cap in re.captures_iter(text) {
            array.push(&JsValue::from_str(&cap[0]));
        }
        Ok(array)
    }

    #[wasm_bindgen]
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
    pub fn markdown_to_html(&self, markdown: &str) -> String {
        let parser = pulldown_cmark::Parser::new(markdown);
        let mut html_output = String::new();
        pulldown_cmark::html::push_html(&mut html_output, parser);
        html_output
    }

    #[wasm_bindgen]
    pub fn base64_encode(&self, text: &str) -> String {
        general_purpose::STANDARD.encode(text)
    }

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if decoding fails.
    pub fn base64_decode(&self, encoded: &str) -> Result<String, JsValue> {
        match general_purpose::STANDARD.decode(encoded) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(string) => Ok(string),
                Err(e) => Err(JsValue::from_str(&format!("Invalid UTF-8 sequence: {e}"))),
            },
            Err(e) => Err(JsValue::from_str(&format!("Invalid base64 string: {e}"))),
        }
    }

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if json fails to parse.
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

    #[wasm_bindgen]
    pub fn chunk_text(&self, text: &str, chunk_size: usize, overlap: usize) -> js_sys::Array {
        let array = js_sys::Array::new();
        if chunk_size == 0 { return array; }

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut i = 0;

        while i < words.len() {
            let end = std::cmp::min(i + chunk_size, words.len());
            let chunk = words[i..end].join(" ");
            array.push(&JsValue::from_str(&chunk));

            if overlap >= chunk_size || end == words.len() { break; }
            i += chunk_size - overlap;
        }
        array
    }

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if lengths do not match or empty.
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

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if regex fails.
    pub fn mask_pii(&self, text: &str) -> Result<String, JsValue> {
        let email_re = get_or_init_regex!(EMAIL_RE, r"([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")?;
        let mut masked = email_re.replace_all(text, "[EMAIL REDACTED]").to_string();

        let phone_re = get_or_init_regex!(PHONE_RE, r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")?;
        masked = phone_re.replace_all(&masked, "[PHONE REDACTED]").to_string();

        Ok(masked)
    }

    #[wasm_bindgen]
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

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if json fails to parse.
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
    pub fn count_tokens_approx(&self, text: &str, method: &str) -> u32 {
        match method {
            "word" => text.split_whitespace().count() as u32,
            _ => ((text.len() as f32) / 4.0).ceil() as u32,
        }
    }

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if regex fails.
    pub fn extract_code_blocks(&self, markdown: &str) -> Result<js_sys::Array, JsValue> {
        let re = get_or_init_regex!(CODE_BLOCK_RE, r"```(\w*)\n([\s\S]*?)```")?;
        let array = js_sys::Array::new();
        for cap in re.captures_iter(markdown) {
            let lang = cap.get(1).map_or("", |m| m.as_str());
            let code = cap.get(2).map_or("", |m| m.as_str());
            let entry = format!("{lang}|{code}");
            array.push(&JsValue::from_str(&entry));
        }
        Ok(array)
    }

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if json parsing fails.
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

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if json parsing fails.
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

    #[wasm_bindgen]
    pub fn normalize_whitespace(&self, text: &str, mode: &str) -> String {
        let collapsed = text.split_whitespace().collect::<Vec<&str>>().join(" ");
        match mode {
            "preserve_paragraphs" => {
                text.lines()
                    .map(|line| line.split_whitespace().collect::<Vec<&str>>().join(" "))
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<String>>()
                    .join("\n\n")
            },
            _ => collapsed,
        }
    }

    #[wasm_bindgen]
    pub fn truncate_with_ellipsis(&self, text: &str, max_len: usize) -> String {
        if text.len() <= max_len { return text.to_string(); }
        if max_len <= 3 { return text.chars().take(max_len).collect(); }

        let mut result = String::with_capacity(max_len);
        let mut char_count = 0;
        for c in text.chars() {
            if char_count >= max_len - 3 { break; }
            result.push(c);
            char_count += c.len_utf8();
        }
        result.push_str("...");
        result
    }

    #[wasm_bindgen]
    /// # Errors
    /// Return an error if extraction fails.
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
    pub fn is_valid_email(&self, email: &str) -> bool {
        if let Ok(re) = get_or_init_regex!(VALID_EMAIL_RE, r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$") {
            re.is_match(email.trim())
        } else {
            false
        }
    }

    #[wasm_bindgen]
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
    pub fn estimate_reading_time(&self, text: &str) -> f32 {
        let word_count = text.split_whitespace().count() as f32;
        (word_count / 200.0).max(0.0)
    }
}
