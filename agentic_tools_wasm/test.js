const { AgenticTools } = require('./pkg/agentic_tools_wasm.js');

const tools = new AgenticTools();

console.log('--- Agentic Tools Wasm Test ---');

// Test URL extraction
const text = "Here are some links: https://example.com and http://test.org/path";
const urls = tools.extract_urls(text);
console.log('Extracted URLs:', urls);

// Test sentiment analysis
const posText = "This is a really great and awesome tool!";
console.log('Sentiment 1:', tools.analyze_sentiment(posText));

const negText = "I had a terrible and awful experience with this bad error.";
console.log('Sentiment 2:', tools.analyze_sentiment(negText));

// Test Markdown to HTML
const md = "# Hello\n\nThis is **bold** text and a [link](https://example.com).";
const html = tools.markdown_to_html(md);
console.log('Markdown to HTML:', html);

// Test Base64 encode/decode
const msg = "Hello from Wasm!";
const encoded = tools.base64_encode(msg);
console.log('Base64 Encoded:', encoded);
const decoded = tools.base64_decode(encoded);
console.log('Base64 Decoded:', decoded);

// Test JSON key extraction
const jsonStr = '{"name": "Agent", "version": "1.0", "active": true}';
const keys = tools.extract_json_keys(jsonStr);
console.log('JSON Keys:', keys);

console.log('\n--- New Agentic Framework Tools ---');

// Test Chunking
const longText = "This is a long document that we might want to split into smaller chunks so that our agent can process it easily within context windows.";
console.log('Text Chunks (size 5, overlap 2):', tools.chunk_text(longText, 5, 2));

// Test Cosine Similarity
const vecA = new Float32Array([1.0, 0.5, 0.1]);
const vecB = new Float32Array([0.9, 0.4, 0.2]);
console.log('Cosine Similarity:', tools.cosine_similarity(vecA, vecB));

// Test PII Masking
const piiText = "Contact me at jules@example.com or call +1-800-555-0199 for more details.";
console.log('Masked PII:', tools.mask_pii(piiText));

// Test Strip HTML
const htmlText = "<div><p>Hello <b>World</b>!</p><script>alert('xss');</script></div>";
console.log('Stripped HTML:', tools.strip_html(htmlText));

// Test Format Prompt
const template = "You are a {{role}}. Your goal is to {{action}} the user's request. Confidence limit: {{confidence}}.";
const vars = JSON.stringify({
    role: "Helpful Assistant",
    action: "resolve",
    confidence: 0.95
});
console.log('Formatted Prompt:', tools.format_prompt(template, vars));

console.log('\n--- Tool Manager Search Test ---');
const { ToolManager } = require('./pkg/agentic_tools_wasm.js');

const router = new ToolManager();
router.set_threshold(0.5);

const q1 = "Coba hitung ini ya...";
console.log(`Tools for query "${q1}":`, router.search_tools(q1));

const q2 = "bagaimana ramalan cuaca hari ini?";
console.log(`Tools for query "${q2}":`, router.search_tools(q2));

const q3 = "tolong cari di web tentang rust";
console.log(`Tools for query "${q3}":`, router.search_tools(q3));

const q4 = "jam berapa sekarang di sana?";
console.log(`Tools for query "${q4}":`, router.search_tools(q4));


console.log('\n--- Gatekeeper Test ---');
const { Gatekeeper } = require('./pkg/agentic_tools_wasm.js');
const gate = new Gatekeeper();

// Test sanitize prompt
const rawPrompt = "Hello\t\r\n\x00World!";
console.log('Sanitized Prompt:', gate.sanitize_prompt_input(rawPrompt));

// Test context truncation
const longContext = "The quick brown fox jumps over the lazy dog";
console.log('Truncated (15 chars):', gate.safe_truncate_context(longContext, 15));

// Test validate LLM output
const validJSON = "```json\n{\"action\": \"click\", \"target\": \"btn\"}\n```";
try {
    console.log('Validated JSON:', gate.validate_llm_output(validJSON, ["action"]));
} catch (e) {
    console.error('Validation failed unexpectedly:', e);
}

const invalidJSON = "```json\n{\"target\": \"btn\"}\n```";
try {
    gate.validate_llm_output(invalidJSON, ["action"]);
    console.error('Validation should have failed for missing key!');
} catch (e) {
    console.log('Validation correctly rejected missing key:', e);
}

console.log('\n--- Local Memory Manager Test ---');
const { LocalMemoryManager } = require('./pkg/agentic_tools_wasm.js');
const mem = new LocalMemoryManager();

try {
    // This will fail in Node.js because window is not defined. We just want to ensure it handles the missing browser environment safely.
    mem.save_memory("key", "val");
} catch(e) {
    console.log("Memory properly threw error in Node environment:", e);
}
