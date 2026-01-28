//! Rust GPU Chimera Demo Library
//!
//! This library demonstrates running the same Rust code on:
//! - CPU (native Rust)
//! - CUDA (via rust-cuda)
//! - Vulkan (via rust-gpu/SPIR-V)

#![feature(once_cell_try)]

// Feature validation

#[cfg(all(feature = "wgpu", feature = "ash"))]
compile_error!("Cannot enable both 'wgpu' and 'ash' features at the same time");

#[cfg(all(target_os = "macos", feature = "cuda"))]
compile_error!("The 'cuda' feature is not supported on macOS. CUDA requires NVIDIA GPUs and is only available on Linux and Windows");

pub mod error;
pub mod runners;

use error::Result;
use shared::{
    BitonicParams, CommutationParams, MajoranaParams, Pass, SortOrder, SortableKey, Stage,
};

fn log_backend_info(
    host: &str,
    backend: Option<&str>,
    adapter: Option<&str>,
    driver: Option<&str>,
) {
    println!("  Host: {host}");

    if let Some(b) = backend {
        println!("  Backend: {b}");
    }

    if let Some(a) = adapter {
        println!("  Adapter: {a}");
    }

    if let Some(d) = driver {
        if !d.is_empty() {
            println!("  Driver: {d}");
        }
    }
}

#[cfg(feature = "wgpu")]
pub fn wgpu_rotate(
    num_chunks_per_term: usize,
    num_terms: usize,
    data: &mut [u32],
    rotation_angle: f32,
    rotation_op: Vec<u32>,
    coefficient_cutoff: f32,
    num_unpaired_cutoff: u32,
) {
    if let Ok(runner) = futures::executor::block_on(WgpuRunnerMajorana::new()) {
        // Get and log backend info
        let (host, backend, adapter, driver) = runner.backend_info();
        log_backend_info(host, backend, adapter.as_deref(), driver.as_deref());

        let len = data.len();

        runner.rotate(
            &mut data[..],
            rotation_angle,
            rotation_op,
            num_terms,
            num_chunks_per_term,
            coefficient_cutoff,
            num_unpaired_cutoff,
        );
    } else if let Err(e) = futures::executor::block_on(WgpuRunnerMajorana::new()) {
        eprintln!("  wgpu initialization failed: {e}");
    }
}

pub fn rotate(
    num_chunks_per_term: usize,
    num_terms: usize,
    data: &mut [u32],
    rotation_angle: f32,
    rotation_op: Vec<u32>,
    coefficient_cutoff: f32,
    num_unpaired_cutoff: u32,
) {
    #[cfg(feature = "wgpu")]
    wgpu_rotate(
        num_chunks_per_term,
        num_terms,
        data,
        rotation_angle,
        rotation_op,
        coefficient_cutoff,
        num_unpaired_cutoff,
    );
}

/// Common trait for all sorting backends
pub trait SortRunner {
    /// Get backend information for logging
    ///
    /// Returns a tuple of (host, backend, adapter, driver)
    fn backend_info(
        &self,
    ) -> (
        &'static str,
        Option<&'static str>,
        Option<String>,
        Option<String>,
    );

    /// Execute a single kernel pass - platform-specific implementation required
    ///
    /// # Arguments
    /// * `data` - The data slice to sort in-place
    /// * `params` - Bitonic sort parameters for this pass
    fn execute_kernel_pass(&self, data: &mut [u32], params: BitonicParams) -> Result<()>;

    /// Prepare data by converting to `u32` representation
    fn prepare_data<T: SortableKey>(&self, data: &[T]) -> (Vec<u32>, usize) {
        let gpu_data: Vec<u32> = data.iter().map(|x| x.to_sortable_u32()).collect();
        (gpu_data, data.len())
    }

    /// Pad data to power of 2 size with appropriate sentinel values
    fn pad_data(&self, data: &mut Vec<u32>, original_size: usize, order: SortOrder) {
        let padded_size = original_size.next_power_of_two();
        if padded_size > original_size {
            let sentinel = match order {
                SortOrder::Ascending => u32::MAX,
                SortOrder::Descending => u32::MIN,
            };
            data.resize(padded_size, sentinel);
        }
    }

    /// Run all bitonic sort stages and passes
    fn run_bitonic_stages(&self, data: &mut [u32], order: SortOrder) -> Result<()> {
        let n = data.len() as u32;
        let num_stages = (n as f32).log2() as u32;

        for stage in 0..num_stages {
            for pass in 0..=stage {
                let params = BitonicParams {
                    num_elements: n,
                    stage: Stage::new(stage),
                    pass_of_stage: Pass::new(pass),
                    sort_order: order.into(),
                };
                self.execute_kernel_pass(data, params)?;
            }
        }
        Ok(())
    }

    /// Convert sorted `u32` data back to original type
    fn finalize_data<T: SortableKey>(&self, gpu_data: &[u32], output: &mut [T]) {
        for (i, &val) in gpu_data.iter().take(output.len()).enumerate() {
            output[i] = T::from_sortable_u32(val);
        }
    }

    /// Sort data with specified order (ascending or descending)
    ///
    /// Sorts the given slice in-place using the bitonic sort algorithm.
    /// The data is converted to `u32` for sorting, then converted back.
    fn sort<T: SortableKey + bytemuck::Pod + Send + Sync>(
        &self,
        data: &mut [T],
        order: SortOrder,
    ) -> Result<()> {
        if data.len() <= 1 {
            return Ok(());
        }

        let (mut gpu_data, original_size) = self.prepare_data(data);
        self.pad_data(&mut gpu_data, original_size, order);
        self.run_bitonic_stages(&mut gpu_data, order)?;
        gpu_data.truncate(original_size);
        self.finalize_data(&gpu_data, data);

        Ok(())
    }
}

pub trait MajoranaRunner {
    /// Get backend information for logging
    ///
    /// Returns a tuple of (host, backend, adapter, driver)
    fn backend_info(
        &self,
    ) -> (
        &'static str,
        Option<&'static str>,
        Option<String>,
        Option<String>,
    );

    /// Execute a single kernel pass - platform-specific implementation required
    ///
    /// # Arguments
    /// * `data` - The data slice to sort in-place
    /// * `params` - Majorana sort parameters for this pass
    fn execute_kernel_pass(&self, data: &mut [u32], params: MajoranaParams) -> Result<()>;

    /// Run majorana shit
    fn run_majorana(
        &self,
        data: &mut [u32],
        rotation_angle: f32,
        rotation_op: Vec<u32>,
        num_terms: usize,
        num_chunks_per_term: usize,
        coefficient_cutoff: f32,
        unpaired_cutoff: u32,
    ) -> Result<()> {
        if rotation_op.len() > 10 {
            panic!("Only 320 majorana modes supported currently.")
        }
        let mut op = [0_u32; 10];
        if rotation_op.len() > 10 {
            panic!("Currently only rotation by 160 Fermionic modes is supported.")
        }
        for ix in 0..rotation_op.len() {
            op[ix] = rotation_op[ix];
        }
        println!("sin: {:}", (rotation_angle * 2.0).sin());
        let params = MajoranaParams {
            num_terms: num_terms as u32,
            num_majorana_modes: num_chunks_per_term as u32,
            cos_angle: (rotation_angle * 2.0).cos(),
            sin_angle: (rotation_angle * 2.0).sin(),
            rotation_op: op,
            coefficient_cutoff,
            unpaired_cutoff,
        };
        self.execute_kernel_pass(data, params)?;
        Ok(())
    }

    /// Convert rotated majorana strings back to rust types?
    fn finalize_data(&self, gpu_data: &[u32], output: &mut [u32]) {
        for (i, &val) in gpu_data.iter().take(output.len()).enumerate() {
            output[i] = val;
        }
    }

    fn rotate(
        &self,
        data: &mut [u32],
        rotation_angle: f32,
        rotation_op: Vec<u32>,
        num_terms: usize,
        num_chunks_per_term: usize,
        coefficient_cutoff: f32,
        unpaired_cutoff: u32,
    ) -> Result<()> {
        if data.len() <= 1 {
            return Ok(());
        }

        let original_size = data.len();
        // self.pad_data(data, original_size);
        self.run_majorana(
            data,
            rotation_angle,
            rotation_op,
            num_terms,
            num_chunks_per_term,
            coefficient_cutoff,
            unpaired_cutoff,
        )?;

        Ok(())
    }
}

// Re-export runners for convenience
pub use runners::CpuRunner;

#[cfg(feature = "cuda")]
pub use runners::CudaRunner;

#[cfg(feature = "wgpu")]
pub use runners::WgpuRunnerMajorana;
// pub use runners::WgpuRunner;

#[cfg(feature = "ash")]
pub use runners::AshRunner;

/// Compiled SPIR-V bytecode for the bitonic sort kernel
// #[cfg(any(feature = "wgpu", feature = "ash"))]
// pub const BITONIC_SPIRV: &[u8] = include_bytes!(env!("MAJORANA_KERNEL_SPV_PATH"));

/// Compiled SPIR-V bytecode for the majorana kernel
#[cfg(any(feature = "wgpu"))]
pub const MAJORANA_SPIRV: &[u8] = include_bytes!(env!("MAJORANA_KERNEL_SPV_PATH"));

/// Compiled PTX code for the bitonic sort kernel
#[cfg(feature = "cuda")]
pub const BITONIC_PTX: &str = include_str!(env!("BITONIC_KERNEL_PTX_PATH"));

/// Verify that a slice is sorted in the specified order
#[cfg(test)]
pub fn verify_sorted<T: SortableKey + PartialOrd>(data: &[T], order: SortOrder) -> bool {
    match order {
        SortOrder::Ascending => data.windows(2).all(|w| w[0] <= w[1]),
        SortOrder::Descending => data.windows(2).all(|w| w[0] >= w[1]),
    }
}
