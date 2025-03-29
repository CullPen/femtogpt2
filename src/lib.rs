#![warn(missing_docs)] // Enforce documentation coverage at the warning level

//! # femto-gpt
//!
//! `femto-gpt` is a minimal Rust library designed for defining, training, and
//! performing inference with Generative Pre-trained Transformer (GPT) language models.
//!
//! It aims for simplicity and educational value while providing core functionalities.
//! It primarily targets CPU execution but includes optional GPU support via OpenCL
//! (enabled with the `gpu` feature).
//!
//! ## Key Components:
//!
//! *   [`GPT`]: The main struct representing the GPT model architecture. Use this to
//!     build, train, and run inference.
//! *   [`Tensor`]: A basic multi-dimensional array implementation used for all numerical data.
//! *   [`Graph`]: A trait defining the computation graph operations. The graph handles
//!     automatic differentiation (backpropagation). An implementation like [`CpuGraph`]
//!     provides the actual computation backend.
//! *   [`Optimizer`]: Traits and implementations (like [`Adam`]) for updating model
//!     parameters during training.
//! *   [`Tokenizer`]: Utilities for converting text data into token IDs suitable for the model.
//!     (e.g., [`SimpleTokenizer`])
//! *   [`funcs`]: Lower-level activation functions and tensor operations used internally
//!     by the computation graph nodes.
//!
//! ## Features
//!
//! *   `gpu`: Enables the OpenCL backend (`ocl` crate) for potential GPU acceleration.
//!     Requires OpenCL drivers and runtime to be installed on the system.
//!
//! ## Example
//!
//! ```no_run
//! use femto_gpt::{GPT, Tensor, CpuGraph, Adam, SimpleTokenizer, GptError}; // Using re-exports
//! use rand::thread_rng;
//!
//! # fn main() -> Result<(), GptError> { // Example requires error handling
//! // --- Configuration ---
//! let vocab_size = 50257; // Example: GPT-2 tokenizer size
//! let embedding_degree = 128; // Dimension of embeddings
//! let num_tokens = 64;     // Context window size
//! let num_layers = 6;      // Number of transformer blocks
//! let num_heads = 4;       // Number of attention heads
//! let head_size = embedding_degree / num_heads; // Size of each head's projection
//! let dropout = 0.1;       // Dropout rate
//!
//! // --- Setup ---
//! let mut rng = thread_rng();
//! // Assuming CpuGraph is the default or desired graph implementation
//! let graph = CpuGraph::new();
//! // Assuming a simple tokenizer exists
//! let tokenizer = SimpleTokenizer::new(); // Or load from file, etc.
//! let dataset_text = "Example text data to train on.";
//! let dataset_indices = tokenizer.encode(dataset_text);
//!
//! // --- Model Creation ---
//! let mut model = GPT::new(
//!     &mut rng,
//!     graph,
//!     None, // Use None for batch_size with train_cpu (parallel instances)
//!     vocab_size,
//!     embedding_degree,
//!     num_tokens,
//!     num_layers,
//!     num_heads,
//!     head_size,
//!     dropout,
//! )?; // Propagate potential errors
//!
//! // --- Training (Illustrative) ---
//! let optimizer = Adam::new();
//! let learning_rate_fn = |step: usize| 0.001 * 0.99f32.powi(step as i32); // Decaying LR
//! let num_training_steps = 100;
//! let training_batch_size = 8; // For train_cpu parallelism
//!
//! println!("Starting training...");
//! // model.train_cpu(
//! //     &dataset_indices,
//! //     num_training_steps,
//! //     training_batch_size,
//! //     None, // Optional gradient clipping/accumulation limit
//! //     &optimizer,
//! //     &learning_rate_fn,
//! //     |m| { println!("Step: {}", m.graph.optimizer_step()); Ok(()) }, // Example callback
//! // )?;
//! println!("Training finished.");
//!
//! // --- Inference ---
//! let prompt_text = "Hello";
//! let prompt_indices = tokenizer.encode(prompt_text);
//! let max_new_tokens = 50;
//! let temperature = 0.8;
//!
//! print!("Prompt: {} -> Generated: ", prompt_text);
//! let generated_indices = model.infer(
//!     &mut rng,
//!     &prompt_indices,
//!     max_new_tokens,
//!     temperature,
//!     |token_id| {
//!         print!("{}", tokenizer.decode(&[token_id]));
//!         // std::io::stdout().flush().unwrap(); // Ensure immediate output if needed
//!     },
//! )?;
//! println!("\nInference complete.");
//!
//! # Ok(())
//! # } // end main
//! ```

// --- Module Declarations ---
// Add brief doc comments explaining the purpose of each module.

/// Contains low-level tensor operations and activation functions (e.g., ReLU, Softmax).
pub mod funcs;
/// Defines the main GPT model architecture, training loop, and inference logic.
pub mod gpt;
/// Defines the computation graph trait (`Graph`) and provides implementations (e.g., `CpuGraph`).
/// Handles forward and backward passes (autodiff).
pub mod graph;
/// Contains optimization algorithms (e.g., `Adam`) used to update model weights.
pub mod optimizer;
/// Provides the multi-dimensional array (`Tensor`) implementation and basic tensor operations.
pub mod tensor;
/// Utilities for text tokenization (converting text to/from numerical IDs).
pub mod tokenizer;

// --- Public API Re-exports ---
// Make the most important types easily accessible from the crate root.

pub use gpt::{GPT, GptError}; // Re-export the main model struct and its potential error type
pub use tensor::{Tensor, TensorError, TensorOps}; // Re-export core tensor types/traits
pub use graph::{Graph, GraphError, TensorId, CpuGraph}; // Re-export graph trait, error, ID, and a common impl
pub use optimizer::{Optimizer, Adam, OptimizerState}; // Re-export optimizer trait, common impl, and state
pub use tokenizer::{Tokenizer, SimpleTokenizer}; // Re-export tokenizer trait and a common impl (adjust if needed)
