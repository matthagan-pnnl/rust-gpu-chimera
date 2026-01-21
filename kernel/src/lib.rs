//! Compute kernel for bitonic sort.
//!
//! This demonstrates the same Rust code running on CUDA, Vulkan (SPIR-V), Metal, HLSL,
//! and CPU.
//!

#![cfg_attr(target_arch = "spirv", no_std)]
#![cfg_attr(target_os = "cuda", no_std)]

#[cfg(any(target_arch = "spirv", target_os = "cuda"))]
use shared::{BitonicParams, MajoranaParams};
use shared::{Pass, SortOrder, Stage, ThreadId};

#[cfg(target_arch = "spirv")]
use spirv_std::{glam::UVec3, num_traits::Float, spirv};

#[cfg(target_os = "cuda")]
use cuda_std::{kernel, thread};

const EVEN_BITS_MASK: u32 = 0x55555555;

/// Newtype wrapper for comparison distance
#[derive(Copy, Clone, Debug)]
pub struct ComparisonDistance(u32);

impl ComparisonDistance {
    #[inline]
    fn from_stage_pass(stage: Stage, pass: Pass) -> Self {
        Self(1u32 << (stage.as_u32() - pass.as_u32()))
    }

    #[inline]
    fn find_partner(&self, thread_id: ThreadId) -> ThreadId {
        ThreadId::new(thread_id.as_u32() ^ self.0)
    }
}

/// Represents a comparison pair in the bitonic network
#[derive(Copy, Clone, Debug)]
pub struct ComparisonPair {
    lower: ThreadId,
    upper: ThreadId,
}

impl ComparisonPair {
    #[inline]
    fn try_new(thread_id: ThreadId, partner: ThreadId) -> (bool, Self) {
        let is_valid = partner.as_u32() > thread_id.as_u32();
        let pair = Self {
            lower: thread_id,
            upper: partner,
        };
        (is_valid, pair)
    }

    #[inline]
    fn is_in_bounds(&self, num_elements: u32) -> bool {
        self.upper.as_u32() < num_elements
    }
}

/// Encapsulates the bitonic sort direction logic
#[derive(Copy, Clone, Debug)]
pub struct BitonicDirection {
    block_ascending: bool,
}

impl BitonicDirection {
    #[inline]
    fn from_position(thread_id: ThreadId, stage: Stage, global_order: SortOrder) -> Self {
        let block_size = 2u32 << stage.as_u32();
        let block_index = thread_id.as_u32() / block_size;
        let block_ascending = block_index % 2 == 0;

        Self {
            block_ascending: match global_order {
                SortOrder::Ascending => block_ascending,
                SortOrder::Descending => !block_ascending,
            },
        }
    }

    #[inline]
    fn should_swap<T: PartialOrd>(&self, val_i: T, val_j: T) -> bool {
        if self.block_ascending {
            val_i > val_j
        } else {
            val_i < val_j
        }
    }
}

/// Generic comparison and swap operation
#[inline]
fn compare_and_swap<T>(data: &mut [T], pair: ComparisonPair, direction: BitonicDirection)
where
    T: Copy + PartialOrd,
{
    let i = pair.lower.as_usize();
    let j = pair.upper.as_usize();

    let val_i = data[i];
    let val_j = data[j];

    if direction.should_swap(val_i, val_j) {
        data[i] = val_j;
        data[j] = val_i;
    }
}

#[inline]
pub fn majorana_rotate_step(
    thread_id: ThreadId,
    data: &mut [u32],
    num_terms: u32,
    num_chunks_per_term: u32,
    rotation_op: [u32; 10],
    coefficient_cutoff: f32,
    unpaired_cutoff: u32,
    cos_angle: f32,
    sin_angle: f32,
) {
    // Early exit for out-of-bounds threads
    if thread_id.as_u32() >= num_terms {
        return;
    }
    let start_ix = thread_id.as_usize() * 2 * (num_chunks_per_term as usize + 2);
    let old_op_start_ix = start_ix + 2;
    let mut coeff = (
        f32::from_bits(data[start_ix]),
        f32::from_bits(data[start_ix + 1]),
    );

    if (coeff.0 * coeff.0 + coeff.1 * coeff.1).sqrt() < coefficient_cutoff {
        for ix in start_ix..start_ix + 2 + num_chunks_per_term as usize {
            data[ix] = 0;
        }
        return;
    }
    let mut num_unpaired = 0;
    for ix in start_ix + 2..start_ix + 2 + num_chunks_per_term as usize {
        num_unpaired += (((data[ix] >> 1) ^ data[ix]) & EVEN_BITS_MASK).count_ones();
    }
    if num_unpaired > unpaired_cutoff {
        for ix in start_ix..start_ix + 2 + num_chunks_per_term as usize {
            data[ix] = 0;
        }
        return;
    }
    let mut term_op_weight = 0;
    let mut rotation_op_weight = 0;
    let mut weight_of_product = 0;
    for ix in 0..num_chunks_per_term as usize {
        weight_of_product += (data[old_op_start_ix + ix] & rotation_op[ix]).count_ones();
        term_op_weight += data[old_op_start_ix + ix].count_ones();
        rotation_op_weight += rotation_op[ix].count_ones();
    }
    if (term_op_weight * rotation_op_weight + weight_of_product) % 2 == 0 {
        return;
    }

    // Including the i sin(theta)
    let mut new_coeff = (-1.0 * coeff.1 * sin_angle, coeff.0 * sin_angle);
    coeff.0 *= cos_angle;
    coeff.1 *= cos_angle;
    data[start_ix] = coeff.0.to_bits();
    data[start_ix + 1] = coeff.1.to_bits();

    let new_op_start_ix = start_ix + 4 + num_chunks_per_term as usize;
    let new_coeff_ix = start_ix + 2 + num_chunks_per_term as usize;

    // TODO: Should probably just track the phase of each term in the coefficient, as it stands is not
    // a great idea. Now we are implicitly tracking the phase, or assuming that the phase is fine.

    // data[new_op_start_ix] = 1;
    if (term_op_weight & 2) != 0 {
        new_coeff = (-1.0 * new_coeff.1, new_coeff.0);
        // data[new_op_start_ix] *= 3;
    }

    if (rotation_op_weight & 2) != 0 {
        new_coeff = (-1.0 * new_coeff.1, new_coeff.0);
        // data[new_op_start_ix] *= 5;
    }

    data[new_op_start_ix] = 0;
    let mut prev_chunk_sum = 0;
    let mut loop_counter: u32 = 0;
    for chunk_ix in 0..num_chunks_per_term as usize {
        loop_counter = loop_counter.wrapping_add(100);
        let mut chunk_1 = rotation_op[chunk_ix];
        let chunk_2 = data[old_op_start_ix + chunk_ix];
        while chunk_1 > 0 {
            let lz = chunk_1.trailing_zeros();
            let filter = (1 << lz) - 1;

            let t = (filter & chunk_2).count_ones();

            data[new_op_start_ix] += prev_chunk_sum + t;

            chunk_1 ^= 1 << lz;
        }
        prev_chunk_sum += chunk_2.count_ones();
    }
    let cross_phase_tot = data[new_op_start_ix];
    let mut new_op_weight = 0;
    for ix in 0..num_chunks_per_term as usize {
        let t = data[old_op_start_ix + ix] ^ rotation_op[ix];
        new_op_weight += t.count_ones();
        data[new_op_start_ix + ix] = t;
    }
    if (new_op_weight & 2) != 0 {
        new_coeff = (new_coeff.1, -1.0 * new_coeff.0);
        // data[new_op_start_ix] *= 7;
    }

    if cross_phase_tot % 2 == 1 {
        new_coeff = (new_coeff.1, -1.0 * new_coeff.0);
        // data[new_op_start_ix] *= 11;
    }

    data[new_coeff_ix] = new_coeff.0.to_bits();
    data[new_coeff_ix + 1] = new_coeff.1.to_bits();
}

/// GPU entry point for Vulkan/SPIR-V
#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(256)))]
pub fn majorana_kernel(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32],
    #[spirv(push_constant)] params: &MajoranaParams,
) {
    let thread_id = ThreadId::new(gid.x);

    majorana_rotate_step(
        thread_id,
        data,
        params.num_terms,
        params.num_majorana_modes,
        params.rotation_op,
        params.coefficient_cutoff,
        params.unpaired_cutoff,
        params.cos_angle,
        params.sin_angle,
    );
}

/// Common bitonic sort logic that works on both CUDA and Vulkan
#[inline]
pub fn bitonic_sort_step(
    thread_id: ThreadId,
    data: &mut [u32],
    stage: Stage,
    pass: Pass,
    num_elements: u32,
    sort_order: SortOrder,
) {
    // Early exit for out-of-bounds threads
    if thread_id.as_u32() >= num_elements {
        return;
    }

    // Calculate comparison distance for this pass
    let distance = ComparisonDistance::from_stage_pass(stage, pass);

    // Find comparison partner
    let partner = distance.find_partner(thread_id);

    // Create comparison pair if valid
    let (is_valid, pair) = ComparisonPair::try_new(thread_id, partner);
    if is_valid && pair.is_in_bounds(num_elements) {
        // Determine sort direction for this comparison
        let direction = BitonicDirection::from_position(thread_id, stage, sort_order);

        // Perform the comparison and swap
        compare_and_swap(data, pair, direction);
    }
}

/// GPU entry point for Vulkan/SPIR-V
#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(256)))]
pub fn bitonic_kernel(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32],
    #[spirv(push_constant)] params: &BitonicParams,
) {
    let thread_id = ThreadId::new(gid.x);

    // Convert u32 to SortOrder
    let sort_order = if params.sort_order == 0 {
        SortOrder::Ascending
    } else {
        SortOrder::Descending
    };

    bitonic_sort_step(
        thread_id,
        data,
        params.stage,
        params.pass_of_stage,
        params.num_elements,
        sort_order,
    );
}

/// GPU entry point for CUDA
#[cfg(target_os = "cuda")]
#[kernel]
pub unsafe fn bitonic_kernel(data: *mut u32, params: BitonicParams) {
    let thread_id =
        ThreadId::new(thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x());

    // Create a slice from the raw pointer
    // Safety: The caller must ensure the pointer is valid for num_elements
    let data_slice = core::slice::from_raw_parts_mut(data, params.num_elements as usize);

    // Convert u32 to SortOrder
    let sort_order = if params.sort_order == 0 {
        SortOrder::Ascending
    } else {
        SortOrder::Descending
    };

    bitonic_sort_step(
        thread_id,
        data_slice,
        params.stage,
        params.pass_of_stage,
        params.num_elements,
        sort_order,
    );
}
