#include <cub/device/device_scan.cuh>

#include <common.cuh>

// %PARAM% TUNE_ITEMS_PER_THREAD ipt 7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24
// %PARAM% TUNE_THREADS_PER_BLOCK tpb 128:160:192:224:256:288:320:352:384:416:448:480:512:544:576:608:640:672:704:736:768:800:832:864:896:928:960:992:1024

#if !TUNE_BASE
template <typename AccumT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using ScanPolicyT = cub::AgentScanPolicy<TUNE_THREADS_PER_BLOCK,
                                             TUNE_ITEMS_PER_THREAD,
                                             AccumT,
                                             cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                             cub::LOAD_DEFAULT,
                                             cub::BLOCK_STORE_WARP_TRANSPOSE,
                                             cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = policy_t;
};
#endif

template <typename T>
static void basic(nvbench::state &state, nvbench::type_list<T>)
{
  using accum_t     = T;
  using input_it_t  = const T *;
  using output_it_t = T *;
  using offset_t    = std::int32_t;
  using output_t    = T;
  using init_t      = cub::detail::InputValue<T>;
  using op_t        = cub::Sum;

#if !TUNE_BASE
  using policy_t    = policy_hub_t<accum_t>;
  using dispatch_t =
    cub::DispatchScan<input_it_t, output_it_t, op_t, init_t, offset_t, accum_t, policy_t>;
#else
  using dispatch_t =
    cub::DispatchScan<input_it_t, output_it_t, op_t, init_t, offset_t, accum_t>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);

  gen(seed_t{}, input);

  T *d_input  = thrust::raw_pointer_cast(input.data());
  T *d_output = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  dispatch_t::Dispatch(nullptr,
                       tmp_size,
                       d_input,
                       d_output,
                       op_t{},
                       init_t{T{}},
                       static_cast<int>(input.size()),
                       0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);
  nvbench::uint8_t *d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec([&](nvbench::launch &launch) {
    dispatch_t::Dispatch(thrust::raw_pointer_cast(tmp.data()),
                         tmp_size,
                         d_input,
                         d_output,
                         op_t{},
                         init_t{T{}},
                         static_cast<int>(input.size()),
                         launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(all_value_types))
  .set_name("cub::DeviceScan::ExclusiveSum")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2));

