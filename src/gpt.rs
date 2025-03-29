use crate::funcs::*;
use crate::graph::{Graph, GraphError, TensorId};
use crate::optimizer::{Optimizer, OptimizerState};
use crate::tensor::{GeneralTensor, Tensor, TensorError, TensorOps}; // Added TensorOps for select

use rand::distributions::WeightedIndex; // More robust sampling
use rand::prelude::*; // Includes Rng trait
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error; // For generic error source
use std::fmt; // For custom error display
use std::time::Instant;

// --- Constants ---
// Consider moving these to a config struct or module if they grow numerous
const POS_ENCODE_BASE: f32 = 10000.0;
const FEEDFORWARD_MULTIPLIER: usize = 4;

// --- Custom Error Type ---
// Provides more specific errors than just GraphError/TensorError bubbling up.
#[derive(Debug)]
pub enum GptError {
    Graph(GraphError),
    Tensor(TensorError),
    Io(std::io::Error), // Example: If loading data from files
    InvalidState(String),
    Other(String), // Catch-all for other issues
}

// Implement standard error traits
impl fmt::Display for GptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GptError::Graph(e) => write!(f, "Graph error: {}", e),
            GptError::Tensor(e) => write!(f, "Tensor error: {}", e),
            GptError::Io(e) => write!(f, "I/O error: {}", e),
            GptError::InvalidState(s) => write!(f, "Invalid state: {}", s),
            GptError::Other(s) => write!(f, "GPT error: {}", s),
        }
    }
}

impl Error for GptError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            GptError::Graph(e) => Some(e),
            GptError::Tensor(e) => Some(e),
            GptError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// Allow converting from underlying errors
impl From<GraphError> for GptError {
    fn from(e: GraphError) -> Self {
        GptError::Graph(e)
    }
}

impl From<TensorError> for GptError {
    fn from(e: TensorError) -> Self {
        GptError::Tensor(e)
    }
}

impl From<std::io::Error> for GptError {
    fn from(e: std::io::Error) -> Self {
        GptError::Io(e)
    }
}

// Define Result type alias for convenience
type GptResult<T> = Result<T, GptError>;

// --- Training State ---
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub tensors: HashMap<String, Tensor<f32>>,
    pub optimizer: OptimizerState,
}

// --- GPT Model ---
#[derive(Debug)] // Added Debug derive
pub struct GPT<G: Graph> {
    graph: G,
    num_tokens: usize,
    vocab_size: usize, // Store vocab_size
    embedding_degree: usize, // Store embedding_degree
    token_input: TensorId,
    pos_input: TensorId,
    output: TensorId,
    expected_output: TensorId,
    loss: TensorId,
    pos_input_fixed: Tensor<f32>,
    // Optional: Store batch size if needed outside constructor
    // batch_size: Option<usize>,
}

// --- Helper Functions ---

/// Samples batches from a dataset.
/// Returns tensors for input sequences (xs) and target sequences (ys).
fn sample_dataset<R: Rng>(
    dataset: &[usize],
    batch_size: usize,
    context_size: usize,
    rng: &mut R,
) -> GptResult<(Tensor<usize>, Tensor<usize>)> {
    if dataset.len() <= context_size {
        return Err(GptError::InvalidState(format!(
            "Dataset length ({}) must be greater than context size ({})",
            dataset.len(),
            context_size
        )));
    }

    let mut xs_data: Vec<usize> = Vec::with_capacity(batch_size * context_size);
    let mut ys_data: Vec<usize> = Vec::with_capacity(batch_size * context_size);

    for _ in 0..batch_size {
        // Ensure start index allows taking context_size + 1 elements without wrapping issues immediately
        let max_start_index = dataset.len() - 1; // Can start at the very last element
        let start_index = rng.gen_range(0..=max_start_index);

        // Efficiently get context_size + 1 elements, handling wrapping
        let mut sequence = Vec::with_capacity(context_size + 1);
        for i in 0..(context_size + 1) {
            sequence.push(dataset[(start_index + i) % dataset.len()]);
        }

        xs_data.extend_from_slice(&sequence[0..context_size]);
        ys_data.extend_from_slice(&sequence[1..context_size + 1]);
    }

    let xs = Tensor::raw(&[batch_size, context_size], xs_data)?;
    let ys = Tensor::raw(&[batch_size, context_size], ys_data)?;

    Ok((xs, ys))
}

/// Selects the next token based on the probability distribution output by the model.
/// Applies temperature scaling and samples from the distribution.
fn select_next_token<R: Rng>(
    rng: &mut R,
    logits: &Tensor<f32>, // Expecting raw logits [vocab_size]
    temperature: f32,
) -> GptResult<usize> {
    if temperature <= 0.0 {
        // Argmax for zero/negative temperature
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        // TODO: Tensor API might have argmax - use it if available!
        for (i, &v) in logits.blob().iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        return Ok(max_idx);
    }

    // Apply temperature scaling to logits
    // Tensors should ideally support broadcasting ops, otherwise iterate
    // let scaled_logits = logits / temperature; // Pseudo-code
    let scaled_logits_data: Vec<f32> = logits.blob().iter().map(|&x| x / temperature).collect();
    let scaled_logits_tensor = Tensor::raw(logits.shape(), scaled_logits_data)?; // Recreate tensor

    // Calculate Softmax probabilities
    // Assuming Softmax::new().run expects a GeneralTensor or similar
    // This part is highly dependent on the `funcs` API
    let probabilities_tensor = Softmax::new().run(
        &[&GeneralTensor::Float(scaled_logits_tensor)], // Wrap appropriately
        false, // is_training = false
    )?;

    let probabilities = probabilities_tensor.blob();

    // Use WeightedIndex for efficient and robust sampling
    // Handle potential errors during distribution creation (e.g., NaN probabilities)
    let dist = WeightedIndex::new(probabilities).map_err(|e| GptError::Other(format!("Failed to create sampling distribution: {}", e)))?;

    Ok(dist.sample(rng))
}


/// Generates positional encoding matrix.
fn generate_positional_encoding(num_tokens: usize, embedding_size: usize) -> GptResult<Tensor<f32>> {
    if embedding_size == 0 {
        return Err(GptError::InvalidState(
            "Embedding size cannot be zero for positional encoding.".to_string(),
        ));
    }

    let mut pe_data = vec![0.0; num_tokens * embedding_size];

    for pos in 0..num_tokens {
        for i in 0..embedding_size {
            let k = pos as f32;
            let two_i = (i / 2 * 2) as f32; // Ensures integer division behavior like original

            let inv_freq = 1.0 / (POS_ENCODE_BASE.powf(two_i / embedding_size as f32));

            let angle = k * inv_freq;

            pe_data[pos * embedding_size + i] = if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
        }
    }

    Tensor::raw(&[num_tokens, embedding_size], pe_data).map_err(GptError::from)
}

// --- Type Aliases for Callbacks ---
// Makes function signatures cleaner
type LearningRateFn = dyn Fn(usize) -> f32;
type TrainCallbackFn<'a, G> = dyn Fn(&mut GPT<G>) -> GptResult<()> + 'a;


// --- GPT Implementation ---
impl<G: Graph> GPT<G> {
    /// Creates a new GPT model configuration and computation graph.
    #[allow(clippy::too_many_arguments)] // Acknowledging many args, Builder pattern is alternative
    pub fn new<R: Rng>(
        rng: &mut R,
        mut graph: G,
        batch_size: Option<usize>, // None implies CPU-style batch-parallelism later
        vocab_size: usize,
        embedding_degree: usize,
        num_tokens: usize, // Also context_size
        num_layers: usize,
        num_heads: usize,
        head_size: usize, // Often embedding_degree / num_heads
        dropout: f32,
    ) -> GptResult<Self> {

        // --- Determine Shape Helper ---
        // Handles optional batch dimension cleanly
        let shape_with_batch = |dims: &[usize]| -> Vec<usize> {
             if let Some(bs) = batch_size {
                let mut shape = vec![bs];
                shape.extend_from_slice(dims);
                shape
            } else {
                dims.to_vec()
            }
        };

        // --- Input Placeholders ---
        let token_input_shape = shape_with_batch(&[num_tokens]);
        let token_input = graph.alloc_usize(Tensor::<usize>::zeros(&token_input_shape), "token_input".into())?;
        let expected_output = graph.alloc_usize(Tensor::<usize>::zeros(&token_input_shape), "expected_output".into())?;

        // --- Embeddings ---
        let token_embedding_table = graph.alloc_param(
            rng,
            &[vocab_size, embedding_degree],
            "token_embedding",
        )?;
        let embedded_token_input = graph.call(Embedding::new(), &[token_input, token_embedding_table])?;

        // --- Positional Encoding ---
        // Alloc as non-trainable data node
        let pos_input_fixed = generate_positional_encoding(num_tokens, embedding_degree)?;
        let pos_input = graph.alloc(pos_input_fixed.clone(), false, "pos_input".into())?; // Keep pos_input_fixed for later use

        // Combine token and positional embeddings
        let mut current_input = graph.call(Add::new(), &[embedded_token_input, pos_input])?;
        current_input = graph.call(Dropout::new(dropout), &[current_input])?; // Dropout after embedding sum

        // --- Transformer Blocks ---
        for l in 0..num_layers {
            current_input = Self::add_transformer_block(
                rng,
                &mut graph,
                current_input, // Input to the block
                l,
                embedding_degree,
                num_tokens,
                num_heads,
                head_size,
                dropout,
            )?;
        }

        // --- Final Layer Norm ---
        let final_norm_coeff = graph.alloc_param(rng, &[embedding_degree], &format!("final_norm_coeff"))?;
        let final_norm_bias = graph.alloc_param_zeros(&[embedding_degree], &format!("final_norm_bias"))?;
        let normed_output = graph.call(LayerNorm::new(), &[current_input, final_norm_coeff, final_norm_bias])?;

        // --- Output Linear Layer (Projection to Vocab) ---
        // Sometimes weight tying is used here (same weights as token_embedding_table transposed)
        // If not tying:
        let output_projection_weights = graph.alloc_param(rng, &[embedding_degree, vocab_size], "output_projection_weights")?;
        let output_projection_bias = graph.alloc_param_zeros(&[vocab_size], "output_projection_bias")?;
        let logits = graph.call(MatMul::new(), &[normed_output, output_projection_weights])?;
        let output = graph.call(Add::new(), &[logits, output_projection_bias])?; // Logits output

        // --- Loss Calculation ---
        let loss = graph.call(CrossEntropy::new(), &[output, expected_output])?;

        Ok(Self {
            graph,
            num_tokens,
            vocab_size,
            embedding_degree,
            token_input,
            pos_input,
            output,
            expected_output,
            loss,
            pos_input_fixed,
            // batch_size,
        })
    }

    /// Helper to add a single transformer block.
    #[allow(clippy::too_many_arguments)]
    fn add_transformer_block<R: Rng>(
        rng: &mut R,
        graph: &mut G,
        block_input: TensorId,
        layer_index: usize,
        embedding_degree: usize,
        num_tokens: usize,
        num_heads: usize,
        head_size: usize,
        dropout: f32,
    ) -> GptResult<TensorId> {
        let l = layer_index; // shorter alias

        // --- Pre-Attention Layer Norm ---
        let norm1_coeff = graph.alloc_param(rng, &[embedding_degree], &format!("l{}_norm1_coeff", l))?;
        let norm1_bias = graph.alloc_param_zeros(&[embedding_degree], &format!("l{}_norm1_bias", l))?;
        let normed_input = graph.call(LayerNorm::new(), &[block_input, norm1_coeff, norm1_bias])?;

        // --- Multi-Head Self-Attention ---
        let mut head_outputs = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            // Linear projections for Key, Query, Value
            let k_params = graph.alloc_param(rng, &[embedding_degree, head_size], &format!("l{}_h{}_k_w", l, h))?;
            let q_params = graph.alloc_param(rng, &[embedding_degree, head_size], &format!("l{}_h{}_q_w", l, h))?;
            let v_params = graph.alloc_param(rng, &[embedding_degree, head_size], &format!("l{}_h{}_v_w", l, h))?;

            // Calculate K, Q, V
            let k = graph.call(MatMul::new(), &[normed_input, k_params])?;
            let q = graph.call(MatMul::new(), &[normed_input, q_params])?;
            let v = graph.call(MatMul::new(), &[normed_input, v_params])?;

            // Scaled Dot-Product Attention
            let q_t = graph.call(Transpose::new(), &[q])?; // Reuse transpose if needed elsewhere?
            let kq = graph.call(MatMul::new(), &[k, q_t])?;

            let head_size_sqrt_inv = (head_size as f32).powf(-0.5);
            let kq_scaled = graph.call(Coeff::new(head_size_sqrt_inv), &[kq])?;

            // Apply causal mask (tril) and softmax
            let masked_kq = graph.call(TrilMask::new(num_tokens), &[kq_scaled])?;
            let attn_weights = graph.call(Softmax::new(), &[masked_kq])?;
            let attn_weights_dropped = graph.call(Dropout::new(dropout), &[attn_weights])?;

            // Apply attention weights to Value
            let head_output = graph.call(MatMul::new(), &[attn_weights_dropped, v])?;
            head_outputs.push(head_output);
        }

        // Concatenate heads
        let concatenated_heads = graph.call(Cat::new(), &head_outputs)?; // Check Cat axis? Assumes last dim.

        // Projection layer after attention
        let proj_weights = graph.alloc_param(rng, &[num_heads * head_size, embedding_degree], &format!("l{}_proj_w", l))?;
        let proj_bias = graph.alloc_param_zeros(&[embedding_degree], &format!("l{}_proj_b", l))?;
        let projected_output = graph.call(MatMul::new(), &[concatenated_heads, proj_weights])?;
        let projected_output_biased = graph.call(Add::new(), &[projected_output, proj_bias])?;
        let projected_output_dropped = graph.call(Dropout::new(dropout), &[projected_output_biased])?;

        // Add residual connection (from block input)
        let attn_residual = graph.call(Add::new(), &[block_input, projected_output_dropped])?;


        // --- Feed-Forward Network ---
        // Pre-FFN Layer Norm
        let norm2_coeff = graph.alloc_param(rng, &[embedding_degree], &format!("l{}_norm2_coeff", l))?;
        let norm2_bias = graph.alloc_param_zeros(&[embedding_degree], &format!("l{}_norm2_bias", l))?;
        let normed_attn_residual = graph.call(LayerNorm::new(), &[attn_residual, norm2_coeff, norm2_bias])?;

        // FFN Layer 1 (Expand)
        let ff1_weights = graph.alloc_param(rng, &[embedding_degree, FEEDFORWARD_MULTIPLIER * embedding_degree], &format!("l{}_ff1_w", l))?;
        let ff1_bias = graph.alloc_param_zeros(&[FEEDFORWARD_MULTIPLIER * embedding_degree], &format!("l{}_ff1_b", l))?;
        let ff1_out = graph.call(MatMul::new(), &[normed_attn_residual, ff1_weights])?;
        let ff1_out_biased = graph.call(Add::new(), &[ff1_out, ff1_bias])?;
        let ff1_activated = graph.call(Gelu::new(), &[ff1_out_biased])?; // Using GELU

        // FFN Layer 2 (Contract)
        let ff2_weights = graph.alloc_param(rng, &[FEEDFORWARD_MULTIPLIER * embedding_degree, embedding_degree], &format!("l{}_ff2_w", l))?;
        let ff2_bias = graph.alloc_param_zeros(&[embedding_degree], &format!("l{}_ff2_b", l))?;
        let ff2_out = graph.call(MatMul::new(), &[ff1_activated, ff2_weights])?;
        let ff2_out_biased = graph.call(Add::new(), &[ff2_out, ff2_bias])?;
        let ff2_out_dropped = graph.call(Dropout::new(dropout), &[ff2_out_biased])?;

        // Add residual connection (from input to FFN)
        let ffn_residual = graph.call(Add::new(), &[attn_residual, ff2_out_dropped])?;

        Ok(ffn_residual) // Output of the block
    }


    /// Synchronizes host-side parameter tensors if the graph backend requires it (e.g., GPU).
    /// Fetches the latest parameters from the compute device.
    pub fn sync(&mut self) -> GptResult<()> {
        for p in self.graph.params() {
            self.graph.fetch(p, false)?; // Fetch only the tensor value, not gradient
        }
        Ok(())
    }

    /// Returns the total number of trainable parameters in the model.
    pub fn num_params(&self) -> GptResult<usize> {
        let mut count = 0;
        for p in self.graph.params() {
            // Use get() which returns Result, avoid unwrap()
            let tensor = self.graph.get(p)?.as_float()?; // Assuming params are always float
            count += tensor.size();
        }
        Ok(count)
    }

    /// Sets the model's parameters and optionally the optimizer state from a TrainingState.
    pub fn set_training_state(
        &mut self,
        training_state: TrainingState,
        load_optimizer: bool,
    ) -> GptResult<()> {
        let params = self.graph.params(); // Get params once
        let mut loaded_params = 0;

        for p in params.iter() {
            let name = self.graph.name_of(*p)?;
            if let Some(tensor_data) = training_state.tensors.get(name) {
                self.graph.load(*p, tensor_data)?;
                loaded_params += 1;
            } else {
                // Optional: Warn if a parameter in the graph is not in the state
                eprintln!("Warning: Parameter '{}' not found in training state.", name);
            }
        }

        // Optional: Warn if state contains tensors not used by the current graph
        for name in training_state.tensors.keys() {
             if !params.iter().any(|p| self.graph.name_of(*p).map_or(false, |n| n == name)) {
                 eprintln!("Warning: Tensor '{}' from training state not found in current graph parameters.", name);
             }
        }


        if loaded_params != params.len() {
             eprintln!("Warning: Loaded {} parameters, but graph expects {}.", loaded_params, params.len());
             // Consider returning an error if strict matching is required:
             // return Err(GptError::InvalidState(format!("Parameter mismatch: Loaded {} but expected {}", loaded_params, params.len())));
        }


        if load_optimizer {
            self.graph.set_optimizer_state(&training_state.optimizer)?;
        }
        Ok(())
    }

    /// Retrieves the model's parameters and optimizer state.
    pub fn get_training_state(&self) -> GptResult<TrainingState> {
        let mut tensors = HashMap::new();
        for p in self.graph.params().iter() {
            let name = self.graph.name_of(*p)?.to_string();
            // Clone the tensor data from the graph
            let tensor_data = self.graph.get(*p)?.as_float()?.clone();
            tensors.insert(name, tensor_data);
        }
        let optimizer_state = self.graph.get_optimizer_state()?;

        Ok(TrainingState {
            tensors,
            optimizer: optimizer_state,
        })
    }


    /// Trains the model using CPU-based batch parallelism via Rayon.
    /// Each Rayon thread processes one instance of the batch. Gradients are averaged.
    pub fn train_cpu<O: Optimizer + Sync, F: LearningRateFn + Sync + Send, C: for<'a> TrainCallbackFn<'a, G> + Sync + Send>(
        &mut self,
        dataset: &[usize],
        num_batches: usize, // Total training steps
        batch_size: usize, // Number of sequences processed in parallel via Rayon
        limit: Option<usize>, // Gradient accumulation/clipping limit? (clarify purpose)
        optimizer: &O,
        learning_rate: &F,
        callback: C,
    ) -> GptResult<()>
    where
        G: Clone + Send + Sync, // Graph must be cloneable and thread-safe
    {
        println!(
            "Starting CPU training: {} steps, batch size {}, context size {}",
            num_batches, batch_size, self.num_tokens
        );
        // Pre-load fixed positional encoding into the main graph
        self.graph.load(self.pos_input, &self.pos_input_fixed)?;

        for i in 0..num_batches {
            let step_timer = Instant::now();
            let current_step = self.graph.optimizer_step(); // Get step before parallel section
            let lr = learning_rate(current_step);

            // --- Parallel Batch Processing ---
            // Collect results (graphs with computed gradients, loss per instance)
            let results: Vec<GptResult<(G, f32)>> = (0..batch_size)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    // Clone graph for this instance
                    // Cloning might be expensive depending on graph implementation
                    let mut instance_graph = self.graph.clone();

                    // Sample data for this single instance
                    let (xs, ys) = sample_dataset(dataset, 1, self.num_tokens, &mut rng)?; // batch_size = 1 here

                    // Load data, forward, backward
                    instance_graph.load_usize(self.token_input, &xs)?;
                    instance_graph.load_usize(self.expected_output, &ys)?;
                    // Ensure pos_input is loaded if cloning doesn't preserve it
                    // instance_graph.load(self.pos_input, &self.pos_input_fixed)?;
                    instance_graph.forward(true)?; // Training mode = true
                    instance_graph.zero_grad()?;
                    let instance_loss = instance_graph.backward_all(self.loss, limit)?;

                    Ok((instance_graph, instance_loss))
                })
                .collect(); // Collect results from parallel iterators


            // --- Gradient Aggregation ---
            // Check for errors and extract graphs and losses
            let (graphs, losses): (Vec<_>, Vec<_>) = results
                .into_iter()
                .collect::<Result<Vec<(G, f32)>, GptError>>()? // Propagate errors
                .into_iter()
                .unzip();


            // Average gradients across all instances
            // This involves iterating through parameters and summing/averaging gradients from each graph instance.
            let param_ids = self.graph.params().to_vec(); // Get parameter IDs once
            let avg_grads: Vec<GptResult<(TensorId, Tensor<f32>)>> = param_ids
                .into_par_iter() // Parallelize gradient averaging too
                .map(|param_id| {
                    // Initialize accumulator tensor (ensure correct shape and device if applicable)
                    // Use graph.get(param_id) to get shape/dtype info if needed.
                    // For simplicity, assuming grads are f32 and size is known/consistent.
                    let first_grad = graphs[0].get_grad(param_id)?;
                    let mut avg_grad_tensor = Tensor::<f32>::zeros(first_grad.shape()); // Start with zeros

                    for graph_instance in &graphs {
                        let grad = graph_instance.get_grad(param_id)?;
                        avg_grad_tensor = (&avg_grad_tensor + grad)?; // Use tensor addition
                    }

                    // Divide by batch size to get average
                    avg_grad_tensor = avg_grad_tensor.map_values(|v| v / batch_size as f32);

                    Ok((param_id, avg_grad_tensor))
                })
                .collect(); // Collect averaged gradients

            // --- Update Main Graph ---
            // Load averaged gradients into the main graph
            for avg_grad_result in avg_grads {
                let (id, avg_grad) = avg_grad_result?;
                self.graph.load_grad(id, &avg_grad)?;
            }

            // Calculate average loss for logging
            let avg_loss: f32 = losses.iter().sum::<f32>() / losses.len() as f32;

            // --- Optimizer Step ---
            self.graph.optimize(optimizer, lr)?; // Use the main graph

            // --- Logging & Callback ---
            let elapsed_ms = step_timer.elapsed().as_millis();
            println!(
                "Step: {} Loss: {:.4} LR: {:.6} (Elapsed: {}ms)",
                self.graph.optimizer_step(), // Should be current_step + 1 now
                avg_loss,
                lr,
                elapsed_ms
            );

            // Execute callback periodically
            if i % 10 == 0 { // Or use optimizer_step % 10 == 0
                // Sync graph state if needed before callback (especially if callback involves inference)
                self.sync()?;
                if let Err(e) = callback(self) {
                     eprintln!("Warning: Callback failed at step {}: {}", self.graph.optimizer_step(), e);
                     // Optionally propagate the error: return Err(e);
                 }
            }
        }

        // Final sync after training loop
        self.sync()?;
        println!("CPU training finished.");
        Ok(())
    }

    /// Trains the model, assuming the graph backend handles batching and parallelism (e.g., GPU).
    pub fn train<O: Optimizer, F: LearningRateFn, C: for<'a> TrainCallbackFn<'a, G>>(
        &mut self,
        dataset: &[usize],
        num_batches: usize, // Total training steps
        batch_size: usize, // Batch size handled by the graph backend
        limit: Option<usize>, // Gradient accumulation/clipping limit?
        optimizer: &O,
        learning_rate: &F,
        callback: C,
    ) -> GptResult<()> {
         println!(
            "Starting GPU/Backend training: {} steps, batch size {}, context size {}",
            num_batches, batch_size, self.num_tokens
        );
        // Pre-load fixed positional encoding
        self.graph.load(self.pos_input, &self.pos_input_fixed)?;

        let mut rng = rand::thread_rng(); // RNG for sampling

        for i in 0..num_batches {
            let step_timer = Instant::now();
            let current_step = self.graph.optimizer_step();
            let lr = learning_rate(current_step);

            // Sample a full batch
            let (xs, ys) = sample_dataset(dataset, batch_size, self.num_tokens, &mut rng)?;

            // Load data for the whole batch
            self.graph.load_usize(self.token_input, &xs)?;
            self.graph.load_usize(self.expected_output, &ys)?;

            // Forward pass, backward pass, optimizer step handled by the graph
            self.graph.forward(true)?;
            self.graph.zero_grad()?;
            // Assuming backward_all calculates loss internally or loss is fetched separately
            // Let's assume backward_all returns the loss value for the batch
            let batch_loss = self.graph.backward_all(self.loss, limit)?;
            self.graph.optimize(optimizer, lr)?;

            // --- Logging & Callback ---
            let elapsed_ms = step_timer.elapsed().as_millis();
             println!(
                "Step: {} Loss: {:.4} LR: {:.6} (Elapsed: {}ms)",
                self.graph.optimizer_step(), // Should be current_step + 1
                batch_loss,
                lr,
                elapsed_ms
            );

            // Execute callback periodically (e.g., every 50 steps)
            if i % 50 == 0 { // Or use optimizer_step % 50 == 0
                 if let Err(e) = callback(self) {
                     eprintln!("Warning: Callback failed at step {}: {}", self.graph.optimizer_step(), e);
                     // Optionally propagate the error: return Err(e);
                 }
            }
        }
        println!("GPU/Backend training finished.");
        Ok(())
    }


    /// Generates text sequence based on a prompt using the trained model.
    pub fn infer<R: Rng, F: FnMut(usize) -> ()>( // Callback is FnMut now
        &mut self,
        rng: &mut R,
        prompt: &[usize],
        count: usize, // Number of tokens to generate *after* the prompt
        temperature: f32,
        mut callback: F, // Closure to call for each generated token (including prompt)
    ) -> GptResult<Vec<usize>> {

        if self.num_tokens == 0 {
            return Err(GptError::InvalidState("Context size (num_tokens) cannot be zero for inference.".to_string()));
        }

        // --- Initialization ---
        let mut generated_sequence = prompt.to_vec();
        let mut context = vec![0; self.num_tokens]; // Initialize context window with padding (e.g., 0)

        // Fill context with prompt, truncating if prompt is longer than context window
        let prompt_len = prompt.len().min(self.num_tokens);
        let start_index = self.num_tokens - prompt_len;
        context[start_index..].copy_from_slice(&prompt[..prompt_len]);

        // Load the fixed positional encoding once
        self.graph.load(self.pos_input, &self.pos_input_fixed)?;

        // Callback for the initial prompt tokens
        for &token in prompt.iter() {
            callback(token);
        }

        // --- Generation Loop ---
        for _ in 0..count {
            // Prepare input tensor (shape [1, num_tokens])
            // Use the current context window
            let input_tensor = Tensor::raw(&[1, self.num_tokens], context.clone())?; // Clone context for tensor
            self.graph.load_usize(self.token_input, &input_tensor)?;

            // Run forward pass (inference mode)
            self.graph.forward(false)?; // is_training = false

            // Fetch the output logits
            // Assuming output shape is [batch_size, num_tokens, vocab_size]
            // We need logits for the *last* token position in the context
            let output_tensor_id = self.output;
            // Fetch may not be needed if graph.get provides access after forward
            self.graph.fetch(output_tensor_id, false)?;
            let output_logits_full = self.graph.get(output_tensor_id)?.as_float()?;

            // Extract logits for the last relevant token position
            // Indexing depends heavily on Tensor API: get(batch_idx)?.get(token_idx)?
            // Assuming batch size is 1 here. Need logits for the last token in the *current* context.
            // The relevant position index is `self.num_tokens - 1`.
            // TODO: Verify tensor indexing logic based on `Tensor` API
            let last_token_logits = output_logits_full.get(0)?.get(self.num_tokens - 1)?;


            // Select the next token using the sampling function
            let next_token = select_next_token(rng, &last_token_logits, temperature)?;


            // --- Update Context and Sequence ---
            // Slide context window: remove first token, append new token
            context.remove(0);
            context.push(next_token);

            generated_sequence.push(next_token);

            // Call the callback with the newly generated token
            callback(next_token);
        }

        Ok(generated_sequence)
    }
}


// --- Helper Trait for Graph ---
// Add helper methods to the Graph trait or as extension trait for cleaner param allocation
trait GraphExt: Graph {
     fn alloc_param<R: Rng>(
        &mut self,
        rng: &mut R,
        shape: &[usize],
        name: &str,
    ) -> Result<TensorId, GraphError> {
        // Consider using a more sophisticated initialization (Kaiming, Xavier)
        // Defaulting to rand for now based on original code
        let tensor = Tensor::<f32>::rand(rng, shape);
        self.alloc(tensor, true, name.into()) // Trainable = true
    }

     fn alloc_param_zeros(
         &mut self,
         shape: &[usize],
         name: &str,
     ) -> Result<TensorId, GraphError> {
         let tensor = Tensor::<f32>::zeros(shape);
         self.alloc(tensor, true, name.into()) // Trainable = true
     }
 }

// Implement the extension trait for any type that implements Graph
impl<G: Graph> GraphExt for G {}
