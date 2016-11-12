#include "RandomForest.h"

#include "MLTree.h"

#include <fstream>
#include <numeric>
RandomForest::RandomForest()
{
}


RandomForest::~RandomForest()
{
}

void RandomForest::learn(
    std::vector<DbTuple>& myDB,
    u64 numTrees,
    u64 minSplit,
    std::vector<DbTuple>* evalData)
{
    mTrees.reset(new MLTree[numTrees]);

    double maxY(9999);

    std::vector<DbTuple>* db = &myDB;
    std::vector<DbTuple> updatedDB(myDB);



    for (i64 treeIdx = 0; treeIdx < numTrees; ++treeIdx)
    {

        // learn a  simple decision tree
        mTrees[treeIdx].learn(*db, minSplit, true);

        ++mNumTrees;

        // evalute the current performance of the model
        if (evalData)//&& ((treeIdx % 10 == 0) || treeIdx == numTrees - 1)
        {

            // compute the predictions for each eval example.
            // Compute the L1, L2, and max error.

            double YSq = 0, YSum = 0;
            maxY = 0;

            double correct = 0;
            for (u64 i = 0; i < evalData->size(); i++)
            {
                auto y = (*evalData)[i].mValue;
                //auto yprime = mTrees[treeIdx].evaluate((*evalData)[i]) * learningRate;

                auto yprime  = evaluate((*evalData)[i]);

                auto Lprime = y - yprime;


                if (std::abs(Lprime) < .5) ++correct;
                //(*evalData)[i].mValue = Lprime;

                YSum += std::abs(Lprime);
                YSq += Lprime * Lprime;


                if (std::abs(Lprime) > maxY)
                {
                    maxY = std::abs(Lprime);
                }
            }


            auto w = std::setw(8);
            auto totalDepth = getTotalDepth();
            double l1 = double(YSum) / (*evalData).size();
            double l2 = double(YSq) / (*evalData).size();

            std::cout
                << " dt " << w << totalDepth << " "
                << " t " << w << treeIdx << " "
                << " l1 " << w << l1
                << " l2 " << w << l2
                << " max " << w << maxY 
                << "  " << w << (correct * 100 / evalData->size()) <<"%"<<std::endl;
        }


        for (auto& leaf : mTrees[treeIdx].mLeafNodes)
        {
            leaf->mRows.clear();
        }
    }
}



void RandomForest::test(
    std::vector<DbTuple>& testData,
    double learningRate)
{
    std::vector<DbTuple> updatedTestData(testData);

    u64 YSq, YSum;

    for (i64 treeIdx = 0; treeIdx < mNumTrees; ++treeIdx)
    {
        YSq = YSum = 0;

        for (u64 i = 0; i < updatedTestData.size(); i++)
        {
            auto y = updatedTestData[i].mValue;
            auto yprime = mTrees[treeIdx].evaluate(updatedTestData[i]);

            auto Lprime = y - learningRate * yprime;

            updatedTestData[i].mValue = Lprime;
            //std::cout
            //	<< "  y =" << y
            //	<< "  y' = " << yprime
            //	<< std::endl;


            YSum += std::abs(Lprime);
            YSq += Lprime * Lprime;
        }

        std::cout
            << "=======================================================" << std::endl
            << "avg l1 error " << double(YSum) / testData.size() << std::endl
            << "avg l2 error " << double(YSq) / testData.size() << std::endl;


    }
}

double RandomForest::evaluate(const DbTuple & data)
{
    double y = 0;

    for (i64 treeIdx = 0; treeIdx < mNumTrees; ++treeIdx)
    {
        y += mTrees[treeIdx].evaluate(data);
    }

    return y / mNumTrees;
}

u64 RandomForest::getTotalDepth()
{
    u64 sum(0);
    for (u64 i = 0; i < mNumTrees; ++i)
        sum += mTrees[i].getDepth();


    return sum;
}

