use serde::{Deserialize, Serialize};
use crate::tensor::{Tensor, TensorError, TensorOps};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt; // For Debug trait on GpuOptimizer

/// Represents the persistent state of an optimizer across training steps.
///
/// Includes the current step count and any optimizer-specific tensors
/// (like momentum and variance for Adam).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizerState {
    /// The number of optimization steps taken so far.
    pub step: usize,
    /// Optimizer-specific state tensors, keyed by a unique name (often derived from parameter names).
    pub state: HashMap<String, Tensor<f32>>,
}

/// Configuration needed for executing an optimizer step on a GPU.
#[cfg(feature = "gpu")]
#[derive(Clone, Debug)] // Added Debug derive
pub struct GpuOptimizerKernel {
    /// Map from buffer names (e.g., "param_name_m", "param_name_v") to their required size (number of elements).
    pub required_buffers: HashMap<String, usize>,
    /// The OpenCL kernel source code.
    pub source_code: String,
    /// The name of the kernel function to execute (e.g., "adamw_step").
    pub kernel_name: String,
}

/// Trait for optimization algorithms used to update model parameters.
///
/// Optimizers are typically stateful (e.g., tracking momentum). The state is managed
/// externally via the `OptimizerState` struct.
pub trait Optimizer: Clone + Serialize + serde::de::DeserializeOwned + Send + Sync + fmt::Debug {
    /// Performs a single optimization step.
    ///
    /// # Arguments
    ///
    /// * `params`: A map where keys are parameter names and values are tuples
    ///   containing a mutable reference to the parameter tensor and an immutable reference
    ///   to its corresponding gradient tensor.
    /// * `optimizer_state`: Mutable reference to the optimizer's state (step count, moments, etc.).
    /// * `learning_rate`: The learning rate for this step.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the step was successful, `Err(TensorError)` otherwise.
    fn step(
        &self,
        params: &mut HashMap<String, (Tensor<f32>, &Tensor<f32>)>, // Changed signature slightly for clarity
        optimizer_state: &mut OptimizerState,
        learning_rate: f32,
    ) -> Result<(), TensorError>;

    /// Provides the necessary information to run the optimizer step on a GPU using OpenCL.
    ///
    /// # Arguments
    ///
    /// * `param_shapes`: A map from parameter names to their tensor shapes (needed to calculate buffer sizes).
    ///
    /// # Returns
    ///
    /// A `GpuOptimizerKernel` struct containing kernel source, name, and required buffer info.
    #[cfg(feature = "gpu")]
    fn gpu_kernel_info(&self, param_shapes: &HashMap<String, &[usize]>) -> GpuOptimizerKernel;
}

// Default epsilon value for numerical stability.
const DEFAULT_EPSILON: f32 = 1e-8;

/// AdamW Optimizer Implementation.
///
/// Implements the AdamW algorithm (Adam with decoupled weight decay).
/// Reference: "Decoupled Weight Decay Regularization" (https://arxiv.org/abs/1711.05101)
/// PyTorch implementation: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamW {
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub epsilon: f32,
}

impl AdamW {
    /// Creates a new AdamW optimizer with default hyperparameters.
    ///
    /// Default values:
    /// - beta1: 0.9
    /// - beta2: 0.999
    /// - weight_decay: 0.01
    /// - epsilon: 1e-8
    pub fn new(beta1: f32, beta2: f32, weight_decay: f32, epsilon: f32) -> Self {
        Self { beta1, beta2, weight_decay, epsilon }
    }
}

impl Default for AdamW {
    /// Creates a new AdamW optimizer with default hyperparameters.
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            epsilon: DEFAULT_EPSILON,
        }
    }
}

impl Optimizer for AdamW {
    fn step(
        &self,
        params: &mut HashMap<String, (Tensor<f32>, &Tensor<f32>)>, // Use &mut HashMap
        optimizer_state: &mut OptimizerState,
        learning_rate: f32,
    ) -> Result<(), TensorError> {
        let current_step = optimizer_state.step + 1; // Use 1-based step for bias correction

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(current_step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(current_step as i32);

        // We collect the updated state tensors because we cannot easily mutate
        // `optimizer_state.state` and `params` inside the parallel iterator
        // due to borrowing rules.
        let updated_states: Vec<(String, Tensor<f32>, Tensor<f32>)> = params
            .par_iter_mut() // Parallel iteration over mutable params
            .map(|(name, (param, grad))| {
                // Generate keys for state tensors
                let m_key = format!("{}_m", name);
                let v_key = format!("{}_v", name);

                // Get or initialize momentum (m) and variance (v) states
                // Use entry API for cleaner initialization
                let m = optimizer_state
                    .state
                    .entry(m_key.clone())
                    .or_insert_with(|| Tensor::zeros(param.shape()));
                let v = optimizer_state
                    .state
                    .entry(v_key.clone())
                    .or_insert_with(|| Tensor::zeros(param.shape()));

                // --- AdamW Update Steps ---

                // 1. Apply weight decay directly to the parameter
                // param = param - param * learning_rate * weight_decay
                // Note: This modifies `param` in place.
                *param = (&*param - ¶m.map_values(|p| p * learning_rate * self.weight_decay))?;


                // 2. Update biased first moment estimate (m)
                // m = beta1 * m + (1 - beta1) * grad
                let m_update = (m.map_values(|val| val * self.beta1))?
                    + &(grad.map_values(|g| g * (1.0 - self.beta1)))?;


                // 3. Update biased second raw moment estimate (v)
                // v = beta2 * v + (1 - beta2) * grad^2
                let grad_sq = grad * grad?; // Calculate grad^2
                let v_update = (v.map_values(|val| val * self.beta2))?
                    + &(grad_sq.map_values(|gsq| gsq * (1.0 - self.beta2)))?;


                // 4. Compute bias-corrected moment estimates
                // m_hat = m / (1 - beta1^step)
                // v_hat = v / (1 - beta2^step)
                // Avoid division by zero if bias_correction is somehow zero
                let m_hat = m_update.map_values(|val| val / bias_correction1)?;
                let v_hat = v_update.map_values(|val| val / bias_correction2)?;

                // 5. Update parameter
                // param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
                let denom = v_hat.map_values(|val| val.sqrt() + self.epsilon)?;
                let update_step = (&m_hat * &denom.map_values(|d| learning_rate / d)?)?; // Equivalent to lr * m_hat / denom

                *param = (&*param - &update_step)?;

                // Return the updated state tensors to be stored later
                Ok((name.clone(), m_update, v_update))
            })
            .collect::<Result<Vec<_>, TensorError>>()?; // Collect results or propagate error

        // Update the optimizer state outside the parallel loop
        for (name, m_updated, v_updated) in updated_states {
             let m_key = format!("{}_m", name);
             let v_key = format!("{}_v", name);
             // Note: Using entry().or_insert() again handles the case where a param might be new
             // Alternatively, if params are guaranteed to exist from the parallel section:
             optimizer_state.state.insert(m_key, m_updated);
             optimizer_state.state.insert(v_key, v_updated);
         }

        optimizer_state.step = current_step; // Increment step count
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn gpu_kernel_info(&self, param_shapes: &HashMap<String, &[usize]>) -> GpuOptimizerKernel {
        // Kernel source with placeholders for hyperparameters passed as arguments
        let source_code = format!(r#"
        __kernel void adamw_step(__global float *param,
                                 __global float *grad,
                                 __global float *m,    // 1st moment buffer
                                 __global float *v,    // 2nd moment buffer
                                 float learning_rate,
                                 float beta1,
                                 float beta2,
                                 float weight_decay,
                                 float epsilon,
                                 ulong step,         // Use ulong for step
                                 ulong n)            // Total number of elements
        {{
            ulong id = get_global_id(0); // Use ulong for large tensors
            if(id < n) {{
                // Apply weight decay
                param[id] = param[id] * (1.0 - learning_rate * weight_decay);

                // Update moments
                m[id] = beta1 * m[id] + (1.0 - beta1) * grad[id];
                v[id] = beta2 * v[id] + (1.0 - beta2) * grad[id] * grad[id];

                // Bias correction (step is 1-based here, passed from host)
                float bias_correction1 = 1.0 - pow(beta1, (float)step);
                float bias_correction2 = 1.0 - pow(beta2, (float)step);

                float m_hat = m[id] / bias_correction1;
                float v_hat = v[id] / bias_correction2;

                // Parameter update
                param[id] = param[id] - learning_rate * m_hat / (sqrt(v_hat) + epsilon);
            }}
        }}"#); // Use raw string literal `r#""#` for easier multiline handling

        // Calculate required buffer sizes for m and v states
        let required_buffers = param_shapes
            .iter()
            .flat_map(|(k, shape)| {
                // Calculate total number of elements in the tensor
                let size = shape.iter().product();
                // Generate keys and sizes for m and v buffers
                vec![(format!("{}_m", k), size), (format!("{}_v", k), size)]
            })
            .collect();

        GpuOptimizerKernel {
            source_code,
            required_buffers,
            kernel_name: "adamw_step".into(),
        }
    }
}
