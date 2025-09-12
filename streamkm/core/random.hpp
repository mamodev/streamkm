#pragma once

#include <random>
#include "./xorshift128p.hpp"



#if !defined(STREAMKM_RANDOM_URBG)
    #define STREAMKM_RANDOM_URBG xorshift128plus
#endif

namespace streamkm
{
    using URBG = STREAMKM_RANDOM_URBG;
}