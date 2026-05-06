use wasm_bindgen::prelude::*;
use web_sys::Storage;

const ERR_NO_WINDOW: &str = "Gagal: Objek 'window' tidak ditemukan (Environment bukan Browser)";
const ERR_STORAGE_BLOCKED: &str = "LocalStorage diblokir (Mungkin mode Incognito ketat)";
const ERR_KEY_EMPTY: &str = "Validasi gagal: Key memori tidak boleh kosong";
const ERR_KEY_INVALID: &str = "Tipe kunci tidak valid (harus string)";
const ERR_VALUE_INVALID: &str = "Tipe value tidak valid (harus string)";

#[wasm_bindgen]
#[derive(Default)]
pub struct LocalMemoryManager {}

#[wasm_bindgen]
impl LocalMemoryManager {
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self { Self {} }

    #[inline]
    fn get_storage() -> Result<Storage, JsValue> {
        web_sys::window()
            .ok_or_else(|| JsValue::from_str(ERR_NO_WINDOW))?
            .local_storage()?
            .ok_or_else(|| JsValue::from_str(ERR_STORAGE_BLOCKED))
    }

    #[inline]
    fn validate_key(key: &str) -> Result<(), JsValue> {
        if key.is_empty() {
            return Err(JsValue::from_str(ERR_KEY_EMPTY));
        }
        Ok(())
    }

    /// Save memory to local storage
    ///
    /// # Errors
    /// Returns an error if the environment lacks a window object, local storage is disabled, or key validation fails.
    #[wasm_bindgen]
    pub fn save_memory(&self, key: &str, value: &str) -> Result<(), JsValue> {
        Self::validate_key(key)?;
        Self::get_storage()?.set_item(key, value)
    }

    /// Recall memory from local storage
    ///
    /// # Errors
    /// Returns an error if local storage is inaccessible or key validation fails.
    #[wasm_bindgen]
    pub fn recall_memory(&self, key: &str) -> Result<Option<String>, JsValue> {
        Self::validate_key(key)?;
        Self::get_storage()?.get_item(key)
    }

    /// Forget (remove) a specific key from local storage
    ///
    /// # Errors
    /// Returns an error if local storage is inaccessible.
    #[wasm_bindgen]
    pub fn forget_memory(&self, key: &str) -> Result<bool, JsValue> {
        Self::validate_key(key)?;
        let storage = Self::get_storage()?;
        let existed = storage.get_item(key)?.is_some();
        if existed {
            storage.remove_item(key)?;
        }
        Ok(existed)
    }

    /// Scan and return all keys present in memory
    ///
    /// # Errors
    /// Returns an error if local storage is inaccessible.
    #[wasm_bindgen]
    pub fn scan_memory_keys(&self) -> Result<js_sys::Array, JsValue> {
        let storage = Self::get_storage()?;
        let array = js_sys::Array::new();
        let mut i = 0;

        while let Ok(Some(key)) = storage.key(i) {
            array.push(&JsValue::from_str(&key));
            i += 1;
        }
        Ok(array)
    }

    /// Wipe the entire memory storage
    ///
    /// # Errors
    /// Returns an error if local storage is inaccessible.
    #[wasm_bindgen]
    pub fn wipe_all_memory(&self) -> Result<(), JsValue> {
        Self::get_storage()?.clear()
    }

    /// Batch save with fail-fast validation
    ///
    /// # Errors
    /// Returns an error if input validation fails for any key/value or storage fails mid-write.
    #[wasm_bindgen]
    pub fn save_memory_batch(&self, entries: &js_sys::Object) -> Result<(), JsValue> {
        let storage = Self::get_storage()?;
        let keys = js_sys::Object::keys(entries);
        let mut valid_entries = Vec::with_capacity(keys.length() as usize);

        for key_js in keys.iter() {
            let key_str = key_js.as_string()
                .ok_or_else(|| JsValue::from_str(ERR_KEY_INVALID))?;
            Self::validate_key(&key_str)?;

            let value_js = js_sys::Reflect::get(entries, &key_js)?;
            let value_str = value_js.as_string()
                .ok_or_else(|| JsValue::from_str(ERR_VALUE_INVALID))?;

            valid_entries.push((key_str, value_str));
        }

        for (k, v) in valid_entries {
            storage.set_item(&k, &v)?;
        }
        Ok(())
    }

    /// Recall memory in batches
    ///
    /// # Errors
    /// Returns an error if local storage cannot be read.
    #[wasm_bindgen]
    pub fn recall_memory_batch(&self, keys: &js_sys::Array) -> Result<js_sys::Object, JsValue> {
        let storage = Self::get_storage()?;
        let result = js_sys::Object::new();

        for key_js in keys.iter() {
            let key_str = key_js.as_string()
                .ok_or_else(|| JsValue::from_str(ERR_KEY_INVALID))?;
            Self::validate_key(&key_str)?;

            let value = match storage.get_item(&key_str)? {
                Some(v) => JsValue::from_str(&v),
                None => JsValue::undefined(),
            };
            js_sys::Reflect::set(&result, &key_js, &value)?;
        }
        Ok(result)
    }

    /// Forget (remove) memory in batches
    ///
    /// # Errors
    /// Returns an error if local storage is inaccessible.
    #[wasm_bindgen]
    pub fn forget_memory_batch(&self, keys: &js_sys::Array) -> Result<js_sys::Object, JsValue> {
        let storage = Self::get_storage()?;
        let result = js_sys::Object::new();

        for key_js in keys.iter() {
            let key_str = key_js.as_string()
                .ok_or_else(|| JsValue::from_str(ERR_KEY_INVALID))?;
            Self::validate_key(&key_str)?;

            let existed = storage.get_item(&key_str)?.is_some();
            if existed {
                storage.remove_item(&key_str)?;
            }
            js_sys::Reflect::set(&result, &key_js, &JsValue::from_bool(existed))?;
        }
        Ok(result)
    }
}
