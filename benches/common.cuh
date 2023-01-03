#pragma once

#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_TYPE_STRINGS(__int128_t, "I128", "int128_t");
NVBENCH_DECLARE_TYPE_STRINGS(__uint128_t, "U128", "uint128_t");

using all_value_types = nvbench::
  type_list<nvbench::int8_t, nvbench::int16_t, nvbench::int32_t, nvbench::int64_t, __int128_t>;

