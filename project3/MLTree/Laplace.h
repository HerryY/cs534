#pragma once
//#include "boost/math/distributions/laplace.hpp"
#include <random>
#include "Common/PRNG.h"

class Laplace
{
public:
    Laplace() = delete;
    Laplace(Laplace& c);
    Laplace(u64 seed, double scale = 1);
    ~Laplace();

    double get();


    PRNG mPrng;
    std::exponential_distribution<> d;
    //boost::math::laplace_distribution<> mDist;

};

