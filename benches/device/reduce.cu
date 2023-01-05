#include <cub/device/device_reduce.cuh>

#include <common.cuh>

// %PARAM% TUNE_ITEMS_PER_THREAD ipt 7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24
// %PARAM% TUNE_THREADS_PER_BLOCK tpb 128:160:192:224:256:288:320:352:384:416:448:480:512:544:576:608:640:672:704:736:768:800:832:864:896:928:960:992:1024
// %PARAM% TUNE_ITEMS_PER_VEC_LOAD ipv 1:2:4

#if !TUNE_BASE
template <typename AccumT, typename OffsetT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int threads_per_block  = TUNE_THREADS_PER_BLOCK;
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
#endif

template <typename T, typename OffsetT>
void reduce(nvbench::state &state, nvbench::type_list<T, OffsetT>)
{
  using accum_t     = T;
  using input_it_t  = const T *;
  using output_it_t = T *;
  using offset_t    = typename cub::detail::ChooseOffsetT<OffsetT>::Type;
  using output_t    = T;
  using init_t      = T;
  using op_t        = cub::Sum;
#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t, offset_t>;
  using dispatch_t =
    cub::DispatchReduce<input_it_t, output_it_t, offset_t, op_t, init_t, accum_t, policy_t>;
#else
  using dispatch_t =
    cub::DispatchReduce<input_it_t, output_it_t, offset_t, op_t, init_t, accum_t>;
#endif

  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> in(elements);
  thrust::device_vector<T> out(1);

  gen(seed_t{}, in);

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

NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(all_value_types, offset_types))
  .set_name("cub::DeviceReduce::Reduce")
  .set_type_axes_names({"T", "OffsetT"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

