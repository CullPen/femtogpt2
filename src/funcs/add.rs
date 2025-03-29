use super::Function;
use crate::tensor::{GeneralTensor, Tensor, TensorError, TensorOps}; // TensorOps might not be needed here but good practice

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

/// Represents an element-wise addition operation: `Output = Input1 + Input2`.
///
/// Expects exactly two input tensors of the same shape.
#[derive(Debug, Clone, Default)] // Added Default derive
pub struct Add;

impl Add {
    /// Creates a new Add function node, boxed for the Function trait.
    pub fn new() -> Box<dyn Function> {
        Box::new(Self::default()) // Use default() since it's an empty struct
    }
}

impl Function for Add {
    /// Performs the forward pass: element-wise addition of the two input tensors.
    ///
    /// # Arguments
    /// * `inps`: A slice containing exactly two `GeneralTensor` references (expected to be float tensors).
    /// * `_training`: Ignored for addition.
    ///
    /// # Errors
    /// Returns `TensorError` if:
    /// * Exactly two inputs are not provided.
    /// * Inputs cannot be converted to float tensors.
    /// * Tensor addition fails (e.g., shape mismatch).
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        // Input validation: Ensure exactly two inputs are provided
        if inps.len() != 2 {
            return Err(TensorError::UnexpectedInputCount { // Use a specific error if available
                expected: 2,
                got: inps.len(),
                func_name: "Add".to_string(), // Add context to the error
            });
        }

        // Extract float tensors
        let lhs = inps[0].as_float()?;
        let rhs = inps[1].as_float()?;

        // Perform addition (relies on Tensor's Add implementation for shape checks etc.)
        lhs + rhs
    }

    /// Computes the gradients for the addition operation.
    ///
    /// The gradient of `A + B` w.r.t `A` is `1 * output_gradient`.
    /// The gradient of `A + B` w.r.t `B` is `1 * output_gradient`.
    ///
    /// # Arguments
    /// * `inps`: A slice containing the *original* inputs (ignored for Add gradient calculation, but part of the signature).
    /// * `out_grad`: The gradient flowing back from the next layer.
    ///
    /// # Errors
    /// Returns `TensorError` if the input count doesn't match the expected count during run (consistency check).
    fn grad(
        &self,
        inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        // Optional: Validate input count consistency, though not strictly needed for grad calculation itself
         if inps.len() != 2 {
             return Err(TensorError::UnexpectedInputCount {
                 expected: 2,
                 got: inps.len(),
                 func_name: "Add::grad".to_string(), // Context
             });
         }

        // Both inputs receive the same gradient flowing from the output.
        // Cloning is necessary as the caller expects separate gradient tensors per input.
        // If `Tensor` uses cheap clones (e.g., Arc), this is efficient.
        Ok(vec![out_grad.clone(), out_grad.clone()])
    }

    /// Creates a boxed clone of this function node.
    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    /// Provides GPU-specific implementation details if the `gpu` feature is enabled.
    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        // Delegates to the dedicated GPU implementation for addition.
        // Assumes gpu::add::gpu_impl correctly handles shapes and kernel launch.
        gpu::add::gpu_impl(out_id, inps)
    }
}
