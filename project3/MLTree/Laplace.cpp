#include "Laplace.h"


Laplace::Laplace(Laplace & c)
    : d(c.d),
    mPrng(c.mPrng.get<u64>())
{

}

Laplace::Laplace(u64 seed, double scale)
    :
    mPrng(seed),
    d(1 / scale)

{}


Laplace::~Laplace()
{
}

double Laplace::get()
{
    return (mPrng.getBit() * 2 - 1) * d(mPrng);// / 2;
}
