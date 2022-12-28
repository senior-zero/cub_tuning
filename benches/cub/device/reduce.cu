#pragma once

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <string>

#include <nvbench/nvbench.cuh>

// %PARAM% TUNE_BLOCK_THREADS bt 128:256
// %PARAM% TUNE_ITEMS_PER_THREAD ipt 16:20
// %PARAM% TUNE_ITEMS_PER_VEC_LOAD ipv 1:2:4

template <typename AccumT, typename OffsetT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int threads_per_block  = TUNE_BLOCK_THREADS;
    static constexpr int items_per_thread   = TUNE_ITEMS_PER_THREAD;
    static constexpr int items_per_vec_load = TUNE_ITEMS_PER_VEC_LOAD;

    using ReducePolicy = cub::AgentReducePolicy<threads_per_block,
                                                items_per_thread,
                                                AccumT,
                                                items_per_vec_load,
                                                cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                                cub::LOAD_DEFAULT>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;

    // SegmentedReducePolicy
    using SegmentedReducePolicy = ReducePolicy;
  };

  using MaxPolicy = policy_t;
};

template <typename T>
void reduce(nvbench::state &state, nvbench::type_list<T>)
{
  using accum_t     = T;
  using input_it_t  = const T *;
  using output_it_t = T *;
  using offset_t    = std::int32_t;
  using output_t    = T;
  using init_t      = T;
  using op_t        = cub::Sum;
  using policy_t    = policy_hub_t<accum_t, offset_t>;
  using dispatch_t =
    cub::DispatchReduce<input_it_t, output_it_t, offset_t, op_t, init_t, accum_t, policy_t>;

  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> in(elements);
  thrust::fill(in.begin(), in.begin() + elements / 2, T{1});
  thrust::device_vector<T> out(1);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  // Allocate temporary storage:
  std::size_t temp_size;
  dispatch_t::Dispatch(nullptr,
                       temp_size,
                       d_in,
                       d_out,
                       static_cast<offset_t>(elements),
                       op_t{},
                       init_t{},
                       0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto *temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec([&](nvbench::launch &launch) {
    dispatch_t::Dispatch(temp_storage,
                         temp_size,
                         d_in,
                         d_out,
                         static_cast<offset_t>(elements),
                         op_t{},
                         init_t{},
                         launch.get_stream());
  });
}

using all_value_types =
  nvbench::type_list<nvbench::int8_t, nvbench::int16_t, nvbench::int32_t, nvbench::int64_t, __int128>;

NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(all_value_types))
  .set_name("cub::DeviceReduce::Reduce")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

