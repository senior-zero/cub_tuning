#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <string>

#include <nvbench/nvbench.cuh>
#include <type_traits>

// %PARAM% TUNE_RADIX_BITS bits 5

#if !TUNE_BASE
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_hub_t
{
  constexpr static bool KEYS_ONLY = std::is_same<ValueT, cub::NullType>::value;

  using DominantT = cub::detail::conditional_t<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5,
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5,
      ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t),
      ONESWEEP_RADIX_BITS    = 8,
      OFFSET_64BIT           = sizeof(OffsetT) == 8,
    };

    // Histogram policy
    typedef cub::AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>
      HistogramPolicy;

    // Exclusive sum policy
    typedef cub::AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS> ExclusiveSumPolicy;

    // Onesweep policy
    typedef cub::AgentRadixSortOnesweepPolicy<384,
                                              OFFSET_64BIT && sizeof(KeyT) == 4 && !KEYS_ONLY ? 17
                                                                                              : 21,
                                              DominantT,
                                              1,
                                              cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
                                              cub::BLOCK_SCAN_RAKING_MEMOIZE,
                                              cub::RADIX_SORT_STORE_DIRECT,
                                              ONESWEEP_RADIX_BITS>
      OnesweepPolicy;

    // ScanPolicy
    typedef cub::AgentScanPolicy<512,
                                 23,
                                 OffsetT,
                                 cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                 cub::LOAD_DEFAULT,
                                 cub::BLOCK_STORE_WARP_TRANSPOSE,
                                 cub::BLOCK_SCAN_RAKING_MEMOIZE>
      ScanPolicy;

    // Downsweep policies
    typedef cub::AgentRadixSortDownsweepPolicy<512,
                                               23,
                                               DominantT,
                                               cub::BLOCK_LOAD_TRANSPOSE,
                                               cub::LOAD_DEFAULT,
                                               cub::RADIX_RANK_MATCH,
                                               cub::BLOCK_SCAN_WARP_SCANS,
                                               PRIMARY_RADIX_BITS>
      DownsweepPolicy;
    typedef cub::AgentRadixSortDownsweepPolicy<(sizeof(KeyT) > 1) ? 256 : 128,
                                               47,
                                               DominantT,
                                               cub::BLOCK_LOAD_TRANSPOSE,
                                               cub::LOAD_DEFAULT,
                                               cub::RADIX_RANK_MEMOIZE,
                                               cub::BLOCK_SCAN_WARP_SCANS,
                                               PRIMARY_RADIX_BITS - 1>
      AltDownsweepPolicy;

    // Upsweep policies
    typedef cub::AgentRadixSortUpsweepPolicy<256, 23, DominantT, cub::LOAD_DEFAULT, PRIMARY_RADIX_BITS>
      UpsweepPolicy;
    typedef cub::
      AgentRadixSortUpsweepPolicy<256, 47, DominantT, cub::LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>
        AltUpsweepPolicy;

    // Single-tile policy
    typedef cub::AgentRadixSortDownsweepPolicy<256,
                                               19,
                                               DominantT,
                                               cub::BLOCK_LOAD_DIRECT,
                                               cub::LOAD_LDG,
                                               cub::RADIX_RANK_MEMOIZE,
                                               cub::BLOCK_SCAN_WARP_SCANS,
                                               SINGLE_TILE_RADIX_BITS>
      SingleTilePolicy;
  };

  using MaxPolicy = policy_t;
};
#endif

template <typename T>
void radix_sort_keys(nvbench::state &state, nvbench::type_list<T>)
{
  constexpr bool is_descending   = false;
  constexpr bool is_overwrite_ok = true;

  using key_t    = T;
  using value_t  = cub::NullType;
  using offset_t = std::int32_t;
#if !TUNE_BASE
  using policy_t   = policy_hub_t<key_t, value_t, offset_t>;
  using dispatch_t = cub::DispatchRadixSort<is_descending, key_t, value_t, offset_t, policy_t>;
#else
  using dispatch_t = cub::DispatchRadixSort<is_descending, key_t, value_t, offset_t>;
#endif

  const int begin_bit = 0;
  const int end_bit   = sizeof(key_t) * 8;

  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> buffer_1(elements);
  thrust::device_vector<T> buffer_2(elements);
  thrust::sequence(buffer_1.begin(), buffer_1.end());
  thrust::sequence(buffer_2.begin(), buffer_2.end());

  key_t *d_buffer_1 = thrust::raw_pointer_cast(buffer_1.data());
  key_t *d_buffer_2 = thrust::raw_pointer_cast(buffer_2.data());

  cub::DoubleBuffer<key_t> d_keys(d_buffer_1, d_buffer_2);
  cub::DoubleBuffer<value_t> d_values;

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  // Allocate temporary storage:
  std::size_t temp_size{};
  dispatch_t::Dispatch(nullptr,
                       temp_size,
                       d_keys,
                       d_values,
                       static_cast<offset_t>(elements),
                       begin_bit,
                       end_bit,
                       is_overwrite_ok,
                       0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto *temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec([&](nvbench::launch &launch) {
    dispatch_t::Dispatch(temp_storage,
                         temp_size,
                         d_keys,
                         d_values,
                         static_cast<offset_t>(elements),
                         begin_bit,
                         end_bit,
                         is_overwrite_ok,
                         launch.get_stream());
  });
}

using all_value_types =
  nvbench::type_list<nvbench::int8_t, nvbench::int16_t, nvbench::int32_t, nvbench::int64_t>; // TODO
                                                                                             // __int128

NVBENCH_BENCH_TYPES(radix_sort_keys, NVBENCH_TYPE_AXES(all_value_types))
  .set_name("cub::DeviceRadixSort::SortKeys")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

