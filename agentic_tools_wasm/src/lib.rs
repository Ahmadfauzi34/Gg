#![cfg_attr(
    not(test),
    warn(
        clippy::all,
        clippy::pedantic,
        clippy::cargo,
    )
)]

// ==========================================
// ⛔ STRICT DENY (Keamanan & Anti-Mangkir)
// ==========================================
#![cfg_attr(not(test), deny(
    clippy::correctness,
    clippy::suspicious,
    clippy::unwrap_used,   // Wajib handle error (jangan pakai panics)
    clippy::expect_used,   // Sama seperti unwrap
    clippy::todo,          // Cegah AI/Developer meninggalkan placeholder
    clippy::unimplemented, // Cegah fungsi kosong masuk ke production
))]

// ==========================================
// 🚧 TEMPORARY ALLOW (Tersisa Prioritas Merah & Struktural)
// ==========================================
#![allow(
    clippy::suboptimal_flops,      // 🔴 Paling tinggi: tensor math (Belum dioptimasi)

    // ⚠️ STRUKTURAL: Dipertahankan agar AI tidak merusak arsitektur hot-path SoA
    clippy::too_many_lines,
    clippy::too_many_arguments,
)]

// ==========================================
// 🛡️ PERMANENT ALLOW (Domain Tensor)
// ==========================================
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,

    // [⬡ Carbo] FHRR: i64→f32 cast intentional untuk AVX2 256-bit alignment
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
)]

use wasm_bindgen::prelude::*;
use regex::Regex;
use base64::{Engine as _, engine::general_purpose};

#[wasm_bindgen]
#[derive(Default)]
pub struct AgenticTools {
    // Internal state can be added here
}

#[wasm_bindgen]
impl AgenticTools {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AgenticTools {
        AgenticTools {}
    }

    /// Extract URLs from text
    ///
    /// # Errors
    ///
    /// Will return an error if the underlying regex expression fails to compile.
    #[wasm_bindgen]
    pub fn extract_urls(&self, text: &str) -> Result<js_sys::Array, JsValue> {
        let re = Regex::new(r"https?://[^\s]+")
            .map_err(|e| JsValue::from_str(&format!("Regex compilation error: {e}")))?;

        let array = js_sys::Array::new();

        for cap in re.captures_iter(text) {
            array.push(&JsValue::from_str(&cap[0]));
        }

        Ok(array)
    }

    /// Analyze text sentiment (basic dictionary-based approach)
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

    /// Convert Markdown to HTML
    #[wasm_bindgen]
    pub fn markdown_to_html(&self, markdown: &str) -> String {
        let parser = pulldown_cmark::Parser::new(markdown);
        let mut html_output = String::new();
        pulldown_cmark::html::push_html(&mut html_output, parser);
        html_output
    }

    /// Encode text to Base64
    #[wasm_bindgen]
    pub fn base64_encode(&self, text: &str) -> String {
        general_purpose::STANDARD.encode(text)
    }

    /// Decode Base64 to text
    ///
    /// # Errors
    ///
    /// Will return an error if the input is not a valid base64 sequence,
    /// or if the decoded bytes do not form a valid UTF-8 string.
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
    ///
    /// Will return an error if the input string is not a valid JSON string,
    /// or if the parsed JSON is not an object.
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

    /// Chunk text into overlapping segments (useful for RAG context windows)
    #[wasm_bindgen]
    pub fn chunk_text(&self, text: &str, chunk_size: usize, overlap: usize) -> js_sys::Array {
        let array = js_sys::Array::new();
        if chunk_size == 0 {
            return array;
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut i = 0;

        while i < words.len() {
            let end = std::cmp::min(i + chunk_size, words.len());
            let chunk = words[i..end].join(" ");
            array.push(&JsValue::from_str(&chunk));

            if overlap >= chunk_size || end == words.len() {
                break;
            }
            i += chunk_size - overlap;
        }

        array
    }

    /// Calculate Cosine Similarity between two float arrays (for evaluating embeddings)
    ///
    /// # Errors
    ///
    /// Returns an error if the two arrays have different lengths or if the vectors are empty.
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

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a.sqrt() * norm_b.sqrt()))
    }

    /// Mask Personal Identifiable Information (Emails and basic phone formats)
    ///
    /// # Errors
    ///
    /// Will return an error if the underlying regex expression fails to compile.
    #[wasm_bindgen]
    pub fn mask_pii(&self, text: &str) -> Result<String, JsValue> {
        let email_re = Regex::new(r"([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
            .map_err(|e| JsValue::from_str(&format!("Regex error: {e}")))?;

        let mut masked = email_re.replace_all(text, "[EMAIL REDACTED]").to_string();

        let phone_re = Regex::new(r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
            .map_err(|e| JsValue::from_str(&format!("Regex error: {e}")))?;

        masked = phone_re.replace_all(&masked, "[PHONE REDACTED]").to_string();

        Ok(masked)
    }

    /// Strip HTML tags from a string to get raw text
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

    /// Format a prompt template using variables from a JSON string
    /// Replace `{{key}}` with the corresponding value in the JSON object.
    ///
    /// # Errors
    ///
    /// Will return an error if the variables string is not valid JSON,
    /// or if the parsed JSON is not an object.
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
}
