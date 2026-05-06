use wasm_bindgen::prelude::*;

const ERR_LLM_NOT_JSON: &str = "Gatekeeper: Output LLM bukan format JSON yang dapat diproses";
const ERR_MISSING_KEY: &str = "Gatekeeper: JSON LLM kehilangan key yang diwajibkan";
const ERR_KEY_NOT_STRING: &str = "Gatekeeper: Array required_keys mengandung elemen non-string";
const ERR_ROOT_NOT_OBJECT: &str = "Gatekeeper: Root element JSON harus berupa Object ({...})";

#[wasm_bindgen]
#[derive(Default)]
pub struct Gatekeeper {}

#[wasm_bindgen]
impl Gatekeeper {
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self { Self {} }

    #[inline]
    fn strip_json_markdown(text: &str) -> &str {
        let trimmed = text.trim();

        if let Some(inner) = trimmed.strip_prefix("```json") {
            if let Some(content) = inner.strip_suffix("```") {
                return content.trim();
            }
        }

        if let Some(inner) = trimmed.strip_prefix("```") {
            if let Some(content) = inner.strip_suffix("```") {
                return content.trim();
            }
        }

        trimmed
    }

    /// Validate LLM output JSON strictly with required keys.
    ///
    /// # Errors
    /// Returns an error if the raw response is not valid JSON, doesn't contain a root object,
    /// or misses any of the explicitly required keys.
    #[wasm_bindgen]
    pub fn validate_llm_output(&self, raw_llm_response: &str, required_keys: &js_sys::Array) -> Result<String, JsValue> {
        let clean_str = Self::strip_json_markdown(raw_llm_response);

        let parsed: serde_json::Value = serde_json::from_str(clean_str)
            .map_err(|e| JsValue::from_str(&format!("{ERR_LLM_NOT_JSON}: {e}")))?;

        let map = parsed.as_object()
            .ok_or_else(|| JsValue::from_str(ERR_ROOT_NOT_OBJECT))?;

        for key_js in required_keys.iter() {
            let key_str = key_js.as_string()
                .ok_or_else(|| JsValue::from_str(ERR_KEY_NOT_STRING))?;

            if !map.contains_key(&key_str) {
                return Err(JsValue::from_str(&format!("{ERR_MISSING_KEY}: '{key_str}'")));
            }
        }

        serde_json::to_string(&parsed)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
    }

    /// Sanitize prompt input
    #[wasm_bindgen]
    #[must_use]
    pub fn sanitize_prompt_input(&self, input: &str) -> String {
        input
            .chars()
            .filter(|c| {
                matches!(*c, '\n' | '\t' | '\r' | ' ') || !c.is_control()
            })
            .collect()
    }

    /// Safely truncate text while respecting UTF-8 and whitespace boundaries.
    #[wasm_bindgen]
    #[must_use]
    pub fn safe_truncate_context(&self, text: &str, max_chars: usize) -> String {
        let mut char_count = 0;
        for _ in text.chars() {
            char_count += 1;
            if char_count > max_chars {
                break;
            }
        }
        if char_count <= max_chars {
            return text.to_string();
        }

        let mut result = String::with_capacity(max_chars + 3);
        let mut last_space_byte_len = 0;


        for (current_chars, c) in text.chars().enumerate() {
            if current_chars >= max_chars {
                break;
            }
            if matches!(c, ' ' | '\t' | '\n' | '\r') {
                last_space_byte_len = result.len();
            }
            result.push(c);

        }

        if last_space_byte_len > 0 {
            result.truncate(last_space_byte_len);
        }
        result.push_str("...");
        result
    }
}
