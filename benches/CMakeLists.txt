set(TBM_CTEST_DEVICE "all" CACHE STRING
  "--device <arg> to use when running benchmarks with ctest"
)
set(TBM_CTEST_TIMEOUT "15.0" CACHE STRING
  "--timeout <arg> to use when running benchmarks with ctest (seconds)"
)
set(TBM_CTEST_MAX_NOISE "0.5" CACHE STRING
  "--max-noise <arg> to use when running benchmarks with ctest"
)
option(TBM_CTEST_OUTPUT_MARKDOWN
  "Write markdown logs when running benchmarks with ctest." ON
)
option(TBM_CTEST_OUTPUT_CSV
  "Write csv output when running benchmarks with ctest." ON
)
option(TBM_CTEST_OUTPUT_JSON
  "Write json files when running benchmarks with ctest." ON
)

set(TBM_BENCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(bench.all)

find_package(CUDAToolkit REQUIRED)
add_library(common SHARED common.cu)
target_link_libraries(common PRIVATE CUDA::curand PUBLIC nvbench::main fmt::fmt)
target_include_directories(common PUBLIC "${TBM_BENCH_ROOT}")

function(get_test_params src labels_var defs_var)
  file(READ "${src}" file_data)
  set(param_regex "//[ ]+%PARAM%[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^\n]*)")

  string(REGEX MATCHALL
    "${param_regex}"
    matches
    "${file_data}"
  )

  set(variant_labels)
  set(variant_defs)

  foreach(match IN LISTS matches)
    string(REGEX MATCH
      "${param_regex}"
      unused
      "${match}"
    )

    set(def ${CMAKE_MATCH_1})
    set(label ${CMAKE_MATCH_2})
    set(values "${CMAKE_MATCH_3}")
    string(REPLACE ":" ";" values "${values}")

    # Build lists of test name suffixes (labels) and preprocessor definitions
    # (defs) containing the cartesian product of all param values:
    if (NOT variant_labels)
      foreach(value IN LISTS values)
        list(APPEND variant_labels ${label}_${value})
      endforeach()
    else()
      set(tmp_labels)
      foreach(old_label IN LISTS variant_labels)
        foreach(value IN LISTS values)
          list(APPEND tmp_labels ${old_label}.${label}_${value})
        endforeach()
      endforeach()
      set(variant_labels "${tmp_labels}")
    endif()

    if (NOT variant_defs)
      foreach(value IN LISTS values)
        list(APPEND variant_defs ${def}=${value})
      endforeach()
    else()
      set(tmp_defs)
      foreach(old_def IN LISTS variant_defs)
        foreach(value IN LISTS values)
          list(APPEND tmp_defs ${old_def}:${def}=${value})
        endforeach()
      endforeach()
      set(variant_defs "${tmp_defs}")
    endif()
  endforeach()

  set(${labels_var} "${variant_labels}" PARENT_SCOPE)
  set(${defs_var} "${variant_defs}" PARENT_SCOPE)
endfunction()

function(tbm_add_bench target_name_var bench_name bench_src thrust_target)
  set(bench_target ${bench_name})
  set(${target_name_var} ${bench_target} PARENT_SCOPE)

  set(results_dir "${TBM_BINARY_DIR}/results")
  file(MAKE_DIRECTORY "${results_dir}")

  add_executable(${bench_target} "${bench_src}")
  target_link_libraries(${bench_target} ${thrust_target} common)
  set_target_properties(${bench_target} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${TBM_LIBRARY_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TBM_LIBRARY_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TBM_EXECUTABLE_OUTPUT_DIR}"
  )
  add_dependencies(bench.all ${bench_target})

  set(test_args
    "--device" "${TBM_CTEST_DEVICE}"
    "--timeout" "${TBM_CTEST_TIMEOUT}"
    "--max-noise" "${TBM_CTEST_MAX_NOISE}"
  )
  if (TBM_CTEST_OUTPUT_MARKDOWN)
    list(APPEND test_args --md "${results_dir}/${bench_name}.log.md")
  endif()
  if (TBM_CTEST_OUTPUT_CSV)
    list(APPEND test_args --csv "${results_dir}/${bench_name}.csv")
  endif()
  if (TBM_CTEST_OUTPUT_JSON)
    list(APPEND test_args --json "${results_dir}/${bench_name}.json")
  endif()
  add_test(NAME ${bench_name}
    COMMAND "$<TARGET_FILE:${bench_target}>" "${test_args}"
    COMMAND_EXPAND_LISTS
  )

  # Build a list of labels with each level of the benchmark path. CTest will use
  # these to print a summary of time spent benchmarking each
  # project/algorithm/etc.
  #
  # e.g. for benchmark: bench.cub.device.histogram.even
  # -> bench;bench.cub;bench.cub.device;bench.cub.device.histogram
  set(test_labels)
  # Split benchmark name into list, e.g. bench;cub;device;histogram
  string(REPLACE "." ";." label_flotsam "${bench_name}")
  list(POP_BACK label_flotsam)
  set(test_label "")
  foreach (label_component IN LISTS label_flotsam)
    string(APPEND test_label ${label_component})
    list(APPEND test_labels ${test_label})
  endforeach()
  set_tests_properties(${bench_name} PROPERTIES
    RUN_SERIAL ON
    LABELS "${test_labels}"
  )
endfunction()

function(tbm_add_bench_dir bench_dir)
  foreach(thrust_target IN LISTS THRUST_TARGETS)
    file(GLOB bench_srcs
      CONFIGURE_DEPENDS
      "${bench_dir}/*.cu"
    )

    file(RELATIVE_PATH bench_prefix "${TBM_BENCH_ROOT}" "${bench_dir}")
    file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
    string(REPLACE "/" "." bench_prefix "${bench_prefix}")

    foreach(bench_src IN LISTS bench_srcs)
      get_test_params("${bench_src}" variant_labels variant_defs)
      list(LENGTH variant_labels num_variants)
      math(EXPR range_end "${num_variants} - 1")

      foreach(var_idx RANGE ${range_end})
        get_filename_component(bench_name "${bench_src}" NAME_WLE)
        string(PREPEND bench_name "bench.${bench_prefix}.")
        list(GET variant_labels ${var_idx} label)
        string(APPEND bench_name ".${label}")
        tbm_add_bench(bench_target ${bench_name} "${bench_src}" ${thrust_target})
        list(GET variant_defs ${var_idx} defs)
        string(REPLACE ":" ";" defs "${defs}")
        target_compile_definitions(${bench_target} PRIVATE ${defs})
      endforeach()

      # base tuning
      get_filename_component(bench_name "${bench_src}" NAME_WLE)
      string(PREPEND bench_name "bench.${bench_prefix}.")
      string(APPEND bench_name ".base")
      tbm_add_bench(bench_target ${bench_name} "${bench_src}" ${thrust_target})
      target_compile_definitions(${bench_target} PRIVATE TUNE_BASE=1)

      # target_compile_options(${bench_target} PRIVATE
      #   $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-G -g>
      # )
    endforeach()
  endforeach()
endfunction()

tbm_get_recursive_subdirs(subdirs)

foreach(subdir IN LISTS subdirs)
  tbm_add_bench_dir("${subdir}")
endforeach()
