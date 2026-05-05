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
