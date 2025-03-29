use femto_gpt::gpt::{TrainingState, GPT, GptError}; // Added GptError
#[cfg(not(feature = "gpu"))]
use femto_gpt::graph::CpuGraph;
#[cfg(feature = "gpu")]
use femto_gpt::graph::gpu::GpuGraph;
use femto_gpt::graph::Graph; // Import the trait for type bounds
use femto_gpt::optimizer::AdamW;
use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};

use clap::Parser; // Use clap
use rand::thread_rng; // More idiomatic import
use std::error::Error; // For generic error source
use std::fmt;
use std::fs;
use std::io::{Read, Write}; // Use specific traits
use std::path::PathBuf;
use std::time::Instant; // For timing callbacks

// --- Configuration Structs ---

#[derive(Parser, Debug, Clone)] // Keep config together
struct ModelConfig {
    #[clap(long, default_value = "64", help = "Dimension of token embeddings")]
    embedding_degree: usize,
    #[clap(long, default_value = "64", help = "Context window size (max sequence length)")]
    num_tokens: usize,
    #[clap(long, default_value = "4", help = "Number of transformer layers")]
    num_layers: usize,
    #[clap(long, default_value = "4", help = "Number of attention heads")]
    num_heads: usize,
    #[clap(long, default_value = "0.0", help = "Dropout rate (0.0 = no dropout)")]
    dropout: f32,
}

impl ModelConfig {
    fn head_size(&self) -> Option<usize> {
        if self.embedding_degree % self.num_heads == 0 {
            Some(self.embedding_degree / self.num_heads)
        } else {
            None // Indicate invalid config
        }
    }
}

#[derive(Parser, Debug, Clone)]
struct TrainConfig {
    #[clap(long, default_value = "dataset.txt", help = "Path to the training dataset file")]
    dataset: PathBuf,
    #[clap(long, default_value = "training_state.dat", help = "Path to load/save the model training state")]
    model_path: PathBuf,
    #[clap(long, default_value = "100000", help = "Total number of training steps")]
    num_steps: usize,
    #[clap(
        long,
        default_value = "32",
        help = "Batch size (ignored for CPU training, used for GPU)"
    )]
    batch_size: usize,
    #[clap(long, default_value = "0.001", help = "Base learning rate")]
    base_lr: f32,
    #[clap(long, default_value = "0.00001", help = "Minimum learning rate after decay")]
    min_lr: f32,
    #[clap(long, default_value = "100", help = "Number of warmup steps for learning rate")]
    warmup_steps: usize,
    #[clap(
        long,
        default_value = "50000",
        help = "Number of steps over which to decay learning rate"
    )]
    decay_steps: usize,
    #[clap(
        long,
        default_value = "200",
        help = "Frequency (in steps) for running inference callback and saving model"
    )]
    callback_interval: usize,
    #[clap(
        long,
        help = "Optional limit for backpropagation steps (gradient accumulation simulation)"
    )]
    backward_limit: Option<usize>,
    #[clap(
        long,
        help = "Load only model weights, resetting optimizer state (e.g., for fine-tuning)"
    )]
    reset_optimizer: bool,
}

#[derive(Parser, Debug, Clone)]
struct InferConfig {
    #[clap(long, default_value = "dataset.txt", help = "Path to the dataset file used for tokenization")]
    tokenizer_dataset: PathBuf,
    #[clap(long, default_value = "training_state.dat", help = "Path to load the model training state from")]
    model_path: PathBuf,
    #[clap(long, help = "Initial text prompt to start generation")]
    prompt: String,
    #[clap(long, default_value = "100", help = "Number of new tokens to generate")]
    count: usize,
    #[clap(
        long,
        default_value = "0.7", // Adjusted default for potentially better quality
        help = "Sampling temperature (0 = deterministic argmax, >0 = randomness)"
    )]
    temperature: f32,
}

// --- CLI Commands ---

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
enum Cli {
    /// Train a new model or continue training an existing one
    Train {
        #[clap(flatten)]
        model_config: ModelConfig,
        #[clap(flatten)]
        train_config: TrainConfig,
    },
    /// Generate text using a pre-trained model
    Infer {
        #[clap(flatten)]
        model_config: ModelConfig, // Need model config for inference too
        #[clap(flatten)]
        infer_config: InferConfig,
    },
}

// --- Custom Error Type ---

#[derive(Debug)]
enum AppError {
    Gpt(GptError),
    Io(std::io::Error),
    Bincode(bincode::Error),
    Config(String), // For configuration validation errors
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Gpt(e) => write!(f, "GPT error: {}", e),
            AppError::Io(e) => write!(f, "IO error: {}", e),
            AppError::Bincode(e) => write!(f, "Serialization error: {}", e),
            AppError::Config(s) => write!(f, "Configuration error: {}", s),
        }
    }
}

impl Error for AppError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            AppError::Gpt(e) => Some(e),
            AppError::Io(e) => Some(e),
            AppError::Bincode(e) => Some(e),
            AppError::Config(_) => None,
        }
    }
}

impl From<GptError> for AppError {
    fn from(e: GptError) -> Self {
        AppError::Gpt(e)
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<bincode::Error> for AppError {
    fn from(e: bincode::Error) -> Self {
        AppError::Bincode(e)
    }
}

// --- Helper Functions ---

fn load_tokenizer(path: &PathBuf) -> Result<SimpleTokenizer, AppError> {
    println!("Loading tokenizer dataset from: {:?}", path);
    let dataset_char = fs::read_to_string(path)?;
    Ok(SimpleTokenizer::new(&dataset_char))
}

fn load_training_state(path: &PathBuf) -> Result<Option<TrainingState>, AppError> {
    if path.is_file() {
        println!("Loading training state from: {:?}", path);
        let bytes = fs::read(path)?;
        let ts: TrainingState = bincode::deserialize(&bytes)?;
        Ok(Some(ts))
    } else {
        println!("No existing training state found at: {:?}", path);
        Ok(None)
    }
}

fn save_training_state(path: &PathBuf, state: &TrainingState) -> Result<(), AppError> {
    println!("Saving training state to: {:?}", path);
    let start_time = Instant::now();
    let bytes = bincode::serialize(state)?;
    // Write atomically if possible (write to temp then rename)
    let temp_path = path.with_extension("tmp");
    fs::write(&temp_path, &bytes)?;
    fs::rename(&temp_path, path)?; // Atomic rename
    println!("Saved state in {:.2}s", start_time.elapsed().as_secs_f32());
    Ok(())
}

// Generic function to initialize GPT, abstracting over graph type
fn initialize_gpt<G: Graph>(
    rng: &mut impl rand::Rng,
    mut graph: G, // Take ownership or mutable ref
    config: &ModelConfig,
    vocab_size: usize,
    batch_size_for_gpu: Option<usize>, // Explicitly pass the optional batch size
) -> Result<GPT<G>, AppError> {
    let head_size = config
        .head_size()
        .ok_or_else(|| AppError::Config(format!(
            "Embedding degree ({}) must be divisible by the number of heads ({})",
            config.embedding_degree, config.num_heads
        )))?;

    let gpt = GPT::new(
        rng,
        graph,
        batch_size_for_gpu,
        vocab_size,
        config.embedding_degree,
        config.num_tokens,
        config.num_layers,
        config.num_heads,
        head_size,
        config.dropout,
    )?;
    gpt.sync()?; // Sync after creation (might initialize params on device)
    Ok(gpt)
}

// --- Main Logic ---

fn main() -> Result<(), AppError> {
    let cli = Cli::parse();

    // --- Graph Initialization (Conditional Compilation) ---
    #[cfg(not(feature = "gpu"))]
    let (graph, is_gpu) = (CpuGraph::new(), false);
    #[cfg(feature = "gpu")]
    let (graph, is_gpu) = (GpuGraph::new().map_err(AppError::Gpt)?, true); // Wrap GpuGraph error

    println!(
        "Initialized graph backend: {}",
        if is_gpu { "GPU (OpenCL)" } else { "CPU" }
    );

    // --- Command Handling ---
    match cli {
        Cli::Infer { model_config, infer_config } => {
            run_inference(graph, is_gpu, model_config, infer_config)?;
        }
        Cli::Train { model_config, train_config } => {
            run_training(graph, is_gpu, model_config, train_config)?;
        }
    }

    Ok(())
}

// --- Inference Logic ---
fn run_inference<G: Graph>(
    graph: G,
    is_gpu: bool, // Needed to determine batch size for init
    model_config: ModelConfig,
    infer_config: InferConfig,
) -> Result<(), AppError> {
    let mut rng = thread_rng();

    let tokenizer = load_tokenizer(&infer_config.tokenizer_dataset)?;
    let vocab_size = tokenizer.vocab_size();
    println!("Tokenizer loaded: {} unique tokens", vocab_size);

    // For inference, we typically use batch size 1, even on GPU, unless batching prompts.
    // GPT::new expects an optional batch size if the graph *might* be GPU.
    // We pass None here as inference usually doesn't benefit from large batches in the same way training does.
    let batch_size_arg = if is_gpu { Some(1) } else { None }; // Use 1 for GPU init, None for CPU

    let mut gpt = initialize_gpt(&mut rng, graph, &model_config, vocab_size, batch_size_arg)?;

    // Load the mandatory training state for inference
    let training_state = load_training_state(&infer_config.model_path)?
        .ok_or_else(|| AppError::Config(format!("Training state file not found at {:?}, required for inference.", infer_config.model_path)))?;
    gpt.set_training_state(training_state, true)?; // Load optimizer state too, although not used

    println!("Model loaded with {} parameters.", gpt.num_params()?);
    println!("Generating text with temperature {:.2}...", infer_config.temperature);
    println!("Prompt: {}", infer_config.prompt);
    print!("Output: {}", infer_config.prompt); // Print prompt immediately

    let prompt_tokens = tokenizer.tokenize(&infer_config.prompt);

    let mut output_buffer = String::new(); // Buffer for decoded tokens

    let _generated_tokens = gpt.infer(
        &mut rng,
        &prompt_tokens,
        infer_config.count,
        infer_config.temperature,
        |token_id| {
            let token_str = tokenizer.untokenize(&[token_id]);
            // Print immediately for interactive feel
            print!("{}", token_str);
            let _ = std::io::stdout().flush(); // Ignore flush errors
            output_buffer.push_str(&token_str);
        },
    )?;
    println!(); // Newline after generation

    // Optionally save the full output to a file or process further

    Ok(())
}

// --- Training Logic ---
fn run_training<G: Graph + Clone + Send + Sync>( // Add necessary bounds for train_cpu
    graph: G,
    is_gpu: bool,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Result<(), AppError> {
    let mut rng = thread_rng();

    let tokenizer = load_tokenizer(&train_config.dataset)?;
    let vocab_size = tokenizer.vocab_size();
    println!("Tokenizer loaded: {} unique tokens", vocab_size);

    let dataset_tokens = tokenizer.tokenize(
        &fs::read_to_string(&train_config.dataset)?, // Read dataset content again for tokens
    );
    println!("Dataset loaded: {} tokens", dataset_tokens.len());

    // Pass batch_size only if using GPU
    let batch_size_arg = if is_gpu { Some(train_config.batch_size) } else { None };
    let mut gpt = initialize_gpt(&mut rng, graph, &model_config, vocab_size, batch_size_arg)?;

    // Load optional training state
    if let Some(ts) = load_training_state(&train_config.model_path)? {
        let load_optimizer = !train_config.reset_optimizer;
        println!("Loading existing state (Load Optimizer: {})...", load_optimizer);
        gpt.set_training_state(ts, load_optimizer)?;
        gpt.sync()?; // Ensure state is synced after loading
    }

    println!("Model initialized with {} parameters.", gpt.num_params()?);
    println!("Starting training loop for {} steps...", train_config.num_steps);

    // --- Learning Rate Schedule ---
    let learning_rate = move |step: usize| { // Use move closure
        let lr = if step < train_config.warmup_steps {
            // Linear warmup
            (train_config.base_lr / train_config.warmup_steps as f32) * step as f32
        } else {
            // Cosine decay (or linear decay as originally implemented)
            // Linear decay:
            f32::max(
                train_config.min_lr,
                train_config.base_lr
                    - (train_config.base_lr - train_config.min_lr)
                        * (step - train_config.warmup_steps) as f32
                        / train_config.decay_steps as f32,
            )
            // Example Cosine Decay:
            // train_config.min_lr + 0.5 * (train_config.base_lr - train_config.min_lr) *
            // (1.0 + f32::cos(((step - train_config.warmup_steps) as f32 / train_config.decay_steps as f32) * std::f32::consts::PI))
        };
        // Ensure LR doesn't drop below min_lr, especially with cosine decay calculations
        f32::max(lr, train_config.min_lr)
    };


    // --- Training Callback ---
    // Clone necessary items for the callback closure
    let callback_tokenizer = tokenizer.clone();
    let callback_model_path = train_config.model_path.clone();
    let callback_interval = train_config.callback_interval;

    let callback = move |gpt: &mut GPT<_>| -> Result<(), GptError> { // Return GptError for compatibility
        let current_step = gpt.graph.optimizer_step(); // Assuming this method exists and is correct
        if current_step > 0 && current_step % callback_interval == 0 {
            println!("\n--- Running Callback at Step {} ---", current_step);
            let start_time = Instant::now();
            let mut rng_cb = thread_rng();

            // 1. Run Inference Example
            println!("Generating inference example (temp 0.7)...");
            print!("Output: ");
            let _ = gpt.infer(
                &mut rng_cb,
                &callback_tokenizer.tokenize("\n"), // Start with newline
                100,
                0.7, // Use a fixed temp for callback consistency
                |token_id| {
                     print!("{}", callback_tokenizer.untokenize(&[token_id]));
                     let _ = std::io::stdout().flush();
                 },
            )?;
            println!();

            // 2. Save Model State
            gpt.sync()?; // Ensure graph state is up-to-date before saving
            let ts = gpt.get_training_state()?;
            // Use helper, handle potential error during save
            if let Err(e) = save_training_state(&callback_model_path, &ts) {
                 eprintln!("Warning: Failed to save model state during callback: {}", e);
                 // Decide if this should be a fatal error for training
            }
            println!("Callback finished in {:.2}s", start_time.elapsed().as_secs_f32());
            println!("-----------------------------------\n");
        }
        Ok(())
    };

    // --- Start Training ---
    let optimizer = AdamW::new(); // Using AdamW

    if is_gpu {
        #[cfg(feature = "gpu")]
        {
            println!("Using GPU training path.");
            gpt.train(
                &dataset_tokens,
                train_config.num_steps,
                train_config.batch_size,
                train_config.backward_limit,
                &optimizer,
                learning_rate,
                callback,
            )?;
        }
        #[cfg(not(feature = "gpu"))]
        {
             // This branch should ideally not be reachable if is_gpu is true
             panic!("Inconsistency: is_gpu is true but gpu feature is not enabled.");
         }

    } else {
        #[cfg(not(feature = "gpu"))]
        {
            println!("Using CPU training path (parallel instances). Batch size setting ignored.");
            // For CPU, batch_size dictates Rayon parallelism, not GPU batching
            gpt.train_cpu(
                &dataset_tokens,
                train_config.num_steps,
                train_config.batch_size, // Use batch_size for parallelism level
                train_config.backward_limit,
                &optimizer,
                learning_rate,
                callback,
            )?;
        }
         #[cfg(feature = "gpu")]
         {
             // This branch should ideally not be reachable if is_gpu is false
             panic!("Inconsistency: is_gpu is false but gpu feature is enabled.");
         }
    }

    println!("Training finished.");
    // Final save maybe? Depends on requirements. Callback saves periodically.

    Ok(())
}
