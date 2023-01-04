#include <cub/device/device_radix_sort.cuh>

#include <thrust/device_vector.h>

#include <string>

#include <common.cuh>
#include <type_traits>

// %PARAM% TUNE_RADIX_BITS bits 6:7:8
// %PARAM% TUNE_ITEMS_PER_THREAD ipt 9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24:25
// %PARAM% TUNE_THREADS_PER_BLOCK tpb 96:128:160:192:224:256:288:320:352:384:416:448:480:512:544:576:608:640:672:704:736:768:800:832:864:896:928:960:992:1024

constexpr bool is_descending   = false;
constexpr bool is_overwrite_ok = false;

#if !TUNE_BASE
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_hub_t
{
  constexpr static bool KEYS_ONLY = std::is_same<ValueT, cub::NullType>::value;

  using DominantT = cub::detail::conditional_t<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int ONESWEEP_RADIX_BITS = TUNE_RADIX_BITS;
    static constexpr bool ONESWEEP           = true;
    static constexpr bool OFFSET_64BIT       = sizeof(OffsetT) == 8;

    // Onesweep policy
    using OnesweepPolicy =
      cub::AgentRadixSortOnesweepPolicy<TUNE_THREADS_PER_BLOCK,
                                        TUNE_ITEMS_PER_THREAD,
                                        DominantT,
                                        1,
                                        cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
                                        cub::BLOCK_SCAN_RAKING_MEMOIZE,
                                        cub::RADIX_SORT_STORE_DIRECT,
                                        ONESWEEP_RADIX_BITS>;

    // These kernels are launched once, no point in tuning at the moment
    using HistogramPolicy = cub::AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;
    using ExclusiveSumPolicy = cub::AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;
    using ScanPolicy = cub::AgentScanPolicy<512,
                                            23,
                                            OffsetT,
                                            cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                            cub::LOAD_DEFAULT,
                                            cub::BLOCK_STORE_WARP_TRANSPOSE,
                                            cub::BLOCK_SCAN_RAKING_MEMOIZE>;

    // No point in tuning
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;

    // No point in tuning single-tile policy
    using SingleTilePolicy = cub::AgentRadixSortDownsweepPolicy<256,
                                                                19,
                                                                DominantT,
                                                                cub::BLOCK_LOAD_DIRECT,
                                                                cub::LOAD_LDG,
                                                                cub::RADIX_RANK_MEMOIZE,
                                                                cub::BLOCK_SCAN_WARP_SCANS,
                                                                SINGLE_TILE_RADIX_BITS>;
  };

  using MaxPolicy = policy_t;
};

template <typename KeyT, typename ValueT, typename OffsetT>
constexpr std::size_t max_onesweep_temp_storage_size()
{
  using portion_offset  = int;
  using onesweep_policy = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t::OnesweepPolicy;
  using agent_radix_sort_onesweep_t = cub::
    AgentRadixSortOnesweep<onesweep_policy, is_descending, KeyT, ValueT, OffsetT, portion_offset>;

  return sizeof(typename agent_radix_sort_onesweep_t::TempStorage);
}

template <typename KeyT, typename ValueT, typename OffsetT>
constexpr std::size_t max_temp_storage_size()
{
  using policy_t = typename policy_hub_t<KeyT, ValueT, OffsetT>::policy_t;

  static_assert(policy_t::ONESWEEP);
  return max_onesweep_temp_storage_size<KeyT, ValueT, OffsetT>();
}

template <typename KeyT, typename ValueT, typename OffsetT>
constexpr bool fits_in_default_shared_memory()
{
  return max_temp_storage_size<KeyT, ValueT, OffsetT>() < 48 * 1024;
}
#else
template <typename, typename, typename>
constexpr bool fits_in_default_shared_memory()
{
  return true;
}
#endif

template <typename T, typename OffsetT>
void radix_sort_values(std::integral_constant<bool, true>,
                       nvbench::state &state,
                       nvbench::type_list<T, OffsetT>)
{
  using offset_t = typename cub::detail::ChooseOffsetT<OffsetT>::Type;

  using key_t   = T;
  using value_t = T;
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
  thrust::device_vector<T> keys_buffer_1(elements);
  thrust::device_vector<T> keys_buffer_2(elements);
  thrust::device_vector<T> values_buffer_1(elements);
  thrust::device_vector<T> values_buffer_2(elements);

  key_t *d_keys_buffer_1     = thrust::raw_pointer_cast(keys_buffer_1.data());
  key_t *d_keys_buffer_2     = thrust::raw_pointer_cast(keys_buffer_2.data());
  value_t *d_values_buffer_1 = thrust::raw_pointer_cast(values_buffer_1.data());
  value_t *d_values_buffer_2 = thrust::raw_pointer_cast(values_buffer_2.data());

  gen(seed_t{}, keys_buffer_1);
  gen(seed_t{}, values_buffer_1);

  cub::DoubleBuffer<key_t> d_keys(d_keys_buffer_1, d_keys_buffer_2);
  cub::DoubleBuffer<value_t> d_values(d_values_buffer_1, d_values_buffer_2);

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements * 2, "Size");
  state.add_global_memory_writes<T>(elements * 2);

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
    cub::DoubleBuffer<key_t> keys     = d_keys;
    cub::DoubleBuffer<value_t> values = d_values;

    dispatch_t::Dispatch(temp_storage,
                         temp_size,
                         keys,
                         values,
                         static_cast<offset_t>(elements),
                         begin_bit,
                         end_bit,
                         is_overwrite_ok,
                         launch.get_stream());
  });
}

template <typename T, typename OffsetT>
void radix_sort_values(std::integral_constant<bool, false>,
                       nvbench::state &,
                       nvbench::type_list<T, OffsetT>)
{
  (void)is_descending;
  (void)is_overwrite_ok;
}

template <typename T, typename OffsetT>
void radix_sort_values(nvbench::state &state, nvbench::type_list<T, OffsetT> tl)
{
  using offset_t = typename cub::detail::ChooseOffsetT<OffsetT>::Type;

  radix_sort_values(std::integral_constant<bool, fits_in_default_shared_memory<T, T, offset_t>()>{},
                    state,
                    tl);
}

NVBENCH_BENCH_TYPES(radix_sort_values, NVBENCH_TYPE_AXES(all_value_types, offset_types))
  .set_name("cub::DeviceRadixSort::SortPairs")
  .set_type_axes_names({"T", "OffsetT"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

