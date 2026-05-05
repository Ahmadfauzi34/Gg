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
}
