#pragma once

#include <thrust/device_vector.h>

#include <limits>

#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_TYPE_STRINGS(__int128_t, "I128", "int128_t");
NVBENCH_DECLARE_TYPE_STRINGS(__uint128_t, "U128", "uint128_t");

using all_value_types = nvbench::
  type_list<nvbench::int8_t, nvbench::int16_t, nvbench::int32_t, nvbench::int64_t, __int128_t>;

template <class T>
class value_wrapper_t
{
  T m_val{};

public:
  explicit value_wrapper_t(T val)
      : m_val(val)
  {}

  T get() const { return m_val; }
};

class seed_t : public value_wrapper_t<unsigned long long int>
{
public:
  using value_wrapper_t::value_wrapper_t;

  seed_t()
      : value_wrapper_t(42)
  {}
};

template <typename T>
void gen(seed_t seed,
         thrust::device_vector<T> &data,
         T min = std::numeric_limits<T>::min(),
         T max = std::numeric_limits<T>::max());

