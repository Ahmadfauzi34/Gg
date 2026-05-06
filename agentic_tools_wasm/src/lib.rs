#![cfg_attr(
    not(test),
    warn(
        clippy::all,
        clippy::pedantic,
        clippy::cargo,
        clippy::perf,
        clippy::complexity,
    )
)]

// ==========================================
// ⛔ STRICT DENY (Keamanan & Anti-Mangkir)
// ==========================================
#![cfg_attr(not(test), deny(
    clippy::correctness,
    clippy::suspicious,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::todo,
    clippy::unimplemented,
))]

// ==========================================
// 🚧 TEMPORARY ALLOW & PERMANENT ALLOW
// ==========================================
#![allow(
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
)]

// Deklarasikan modul yang kita pecah
mod agentic_tools;
mod tool_manager;
mod local_memory;
mod gatekeeper;

// Re-export agar bisa diakses langsung oleh Javascript/WASM
pub use agentic_tools::AgenticTools;
pub use tool_manager::ToolManager;
pub use local_memory::LocalMemoryManager;
pub use gatekeeper::Gatekeeper;
