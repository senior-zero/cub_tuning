#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/tabulate.h>

#include <cstdint>

#include <common.cuh>
#include <curand.h>

class generator_t
{
private:
  generator_t();

public:

  static generator_t &instance();
  ~generator_t();

  template <typename T>
  void operator()(seed_t seed,
                  thrust::device_vector<T> &data,
                  T min = std::numeric_limits<T>::min(),
                  T max = std::numeric_limits<T>::max());

  float* distribution();
  curandGenerator_t &gen() { return m_gen; }

  float* prepare_random_generator(
      seed_t seed,
      std::size_t num_items);

private:
  curandGenerator_t m_gen;
  thrust::device_vector<float> m_distribution;
};

generator_t& generator_t::instance()
{
  static generator_t generator;
  return generator;
}

template <typename T>
struct random_to_item_t
{
  float m_min;
  float m_max;

  __host__ __device__ random_to_item_t(T min, T max)
      : m_min(static_cast<float>(min))
      , m_max(static_cast<float>(max))
  {}

  __device__ T operator()(float random_value)
  {
    return static_cast<T>((m_max - m_min) * random_value + m_min);
  }
};

generator_t::generator_t()
{
  curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT);
}

generator_t::~generator_t()
{
  curandDestroyGenerator(m_gen);
}

float* generator_t::distribution()
{
  return thrust::raw_pointer_cast(m_distribution.data());
}

float *generator_t::prepare_random_generator(seed_t seed, 
                                             std::size_t num_items)
{
  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());

  m_distribution.resize(num_items);
  curandGenerateUniform(m_gen,
                        this->distribution(),
                        num_items);

  return this->distribution();
}

template <class T>
void generator_t::operator()(seed_t seed,
                             thrust::device_vector<T> &data,
                             T min,
                             T max)
{
  prepare_random_generator(seed, data.size());

  thrust::transform(m_distribution.begin(),
                    m_distribution.end(),
                    data.begin(),
                    random_to_item_t<T>(min, max));
}

template <typename T>
void gen(seed_t seed, 
         thrust::device_vector<T> &data,
         T min,
         T max)
{
  generator_t::instance()(seed, data, min, max);
}

#define INSTANTIATE_RND(TYPE) \
template \
void gen<TYPE>( \
    seed_t, \
    thrust::device_vector<TYPE> &data, \
    TYPE min, \
    TYPE max)

#define INSTANTIATE(TYPE) \
  INSTANTIATE_RND(TYPE); 

INSTANTIATE(std::uint8_t);
INSTANTIATE(std::uint16_t);
INSTANTIATE(std::uint32_t);
INSTANTIATE(std::uint64_t);
INSTANTIATE(__uint128_t);

INSTANTIATE(std::int8_t);
INSTANTIATE(std::int16_t);
INSTANTIATE(std::int32_t);
INSTANTIATE(std::int64_t);
INSTANTIATE(__int128_t);

