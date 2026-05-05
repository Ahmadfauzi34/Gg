use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use regex::Regex;
use base64::{Engine as _, engine::general_purpose};

#[wasm_bindgen]
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
    #[wasm_bindgen]
    pub fn extract_urls(&self, text: &str) -> js_sys::Array {
        let re = Regex::new(r"https?://[^\s]+").unwrap();
        let array = js_sys::Array::new();

        for cap in re.captures_iter(text) {
            array.push(&JsValue::from_str(&cap[0]));
        }

        array
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

        if score > 0 {
            "positive".to_string()
        } else if score < 0 {
            "negative".to_string()
        } else {
            "neutral".to_string()
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
    #[wasm_bindgen]
    pub fn base64_decode(&self, encoded: &str) -> Result<String, JsValue> {
        match general_purpose::STANDARD.decode(encoded) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(string) => Ok(string),
                Err(e) => Err(JsValue::from_str(&format!("Invalid UTF-8 sequence: {}", e))),
            },
            Err(e) => Err(JsValue::from_str(&format!("Invalid base64 string: {}", e))),
        }
    }

    /// Parse a JSON string and extract keys
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
            Err(e) => Err(JsValue::from_str(&format!("Invalid JSON: {}", e))),
        }
    }
}
