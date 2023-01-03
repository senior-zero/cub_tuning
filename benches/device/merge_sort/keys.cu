#include <cub/device/device_merge_sort.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <string>
#include <type_traits>

#include <common.cuh>

// %PARAM% TUNE_BLOCK_THREADS bt 128:256:512
// %PARAM% TUNE_ITEMS_PER_THREAD ipt 16:17:18:19:20

using value_t  = cub::NullType;
using offset_t = std::int32_t;

#if !TUNE_BASE
template <typename KeyT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using MergeSortPolicy =
      cub::AgentMergeSortPolicy<TUNE_BLOCK_THREADS,
                                cub::Nominal4BItemsToItems<KeyT>(TUNE_ITEMS_PER_THREAD),
                                cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                cub::LOAD_DEFAULT,
                                cub::BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = policy_t;
};
#endif

struct less_t
{
  template <typename DataType>
  __device__ bool operator()(const DataType &lhs, const DataType &rhs)
  {
    return lhs < rhs;
  }
};

template <typename T>
void merge_sort_keys(nvbench::state &state, nvbench::type_list<T>)
{
  using key_t            = T;
  using value_t          = cub::NullType;
  using key_input_it_t   = key_t *;
  using value_input_it_t = value_t *;
  using key_it_t         = key_t *;
  using value_it_t       = value_t *;
  using offset_t         = int;
  using compare_op_t     = less_t;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<key_t>;
  using dispatch_t = cub::DispatchMergeSort<key_input_it_t,
                                            value_input_it_t,
                                            key_it_t,
                                            value_it_t,
                                            offset_t,
                                            compare_op_t,
                                            policy_t>;
#else
  using dispatch_t = 
    cub::DispatchMergeSort<key_input_it_t, value_input_it_t, key_it_t, value_it_t, offset_t, compare_op_t>;
#endif

  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> buffer_1(elements);
  thrust::device_vector<T> buffer_2(elements);
  thrust::sequence(buffer_1.rbegin(), buffer_1.rend());
  thrust::sequence(buffer_2.rbegin(), buffer_2.rend());

  key_t *d_buffer_1 = thrust::raw_pointer_cast(buffer_1.data());
  key_t *d_buffer_2 = thrust::raw_pointer_cast(buffer_2.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  dispatch_t::Dispatch(nullptr,
                       temp_size,
                       d_buffer_1,
                       nullptr,
                       d_buffer_2,
                       nullptr,
                       static_cast<offset_t>(elements),
                       compare_op_t{},
                       0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto *temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec([&](nvbench::launch &launch) {
    dispatch_t::Dispatch(temp_storage,
                         temp_size,
                         d_buffer_1,
                         nullptr,
                         d_buffer_2,
                         nullptr,
                         static_cast<offset_t>(elements),
                         compare_op_t{},
                         launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(merge_sort_keys, NVBENCH_TYPE_AXES(all_value_types))
  .set_name("cub::DeviceMergeSort::SortKeys")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

