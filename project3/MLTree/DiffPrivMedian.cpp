#include "DiffPrivMedian.h"

#include "Laplace.h"
#include <array>

#include "Crypto/PRNG.h"
#include "Common/Defines.h"

DiffPrivMedian::DiffPrivMedian()
    :
    mPrng(0),
    mScaleFrac(1.0 / 3.0)
{
}


DiffPrivMedian::~DiffPrivMedian()
{
}

u64 DiffPrivMedian::getIterations(YType maxValue)
{
    return std::log(maxValue) / std::log(1 / (1 - mScaleFrac));
}


YType DiffPrivMedian::getMedian(std::vector<YType>& myDB, YType maxValue, double epsilon)
{
    auto copy = myDB;

    std::sort(copy.begin(), copy.end());
    return copy[copy.size() / 2];



    auto k = getIterations(maxValue);


    auto factor = std::min(double(k), (2 * std::sqrt(2 *  k * std::log(1 / std::pow(2.0, -20)))));
    auto epsilonPer	= epsilon / factor;

    //std::cout<< "ep " << epsilon << "    ep  per " << epsilonPer << "  = " << epsilon << " / ( 2 * sqrt( 2 * " << k << "  * ln( 1 / pow(2,-20))))" <<  std::endl;
    //auto totalEpsilon
    //	= std::sqrt(2 * k * std::log(1 / std::pow(2.0, -20)))
    //	* epsilonPer + k * epsilonPer * (std::exp(epsilonPer) - 1);


    Laplace lp(mPrng.get<u64>(), 1 / epsilonPer);

    auto iterCount = log2ceil(maxValue);

    YType splitVal = maxValue / 2;
    YType minVal = 0;
    YType maxVal = maxValue;
    YType diff = maxVal - minVal;

    //for (u64 i = 0; i < iterCount; ++i)
    u64 i = 0;
    while (splitVal != minVal)
    {
        
        std::array<YType, 2> counts{ 0,0 };


        for (u64 j = 0; j < myDB.size(); ++j)
        {
            //if(myDB[j].mValue >= minVal && myDB[j].mValue <= maxVal)	
                counts[(myDB[j] > splitVal) & 1]++;
        }

        auto noise = lp.get();

        //std::cout<< "iter " << i << "  " << counts[0] << " + " << noise  << " vs " << counts[1] << std::endl;
         
        //counts[0] += ;

        auto b = counts[0] + noise > counts[1];

        getNextRange(minVal, maxVal, b);


        splitVal = minVal + (maxVal - minVal) / 2;
        ++i;

    }

    return splitVal;

}

void DiffPrivMedian::getNextRange(YType &minVal, YType &maxVal, bool b)
{
    auto diff = maxVal - minVal; 


    auto stepSize =std::max(1.0, diff * mScaleFrac);

    if (b)
    {
        //std::cout<< "   0 " << b << std::endl;
        maxVal -= stepSize;
    }
    else
    {
        //std::cout<< "   1 " << !b << std::endl;
        minVal += stepSize;
    }

}
