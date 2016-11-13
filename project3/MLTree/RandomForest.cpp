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
    PRNG prng(12423524);

    std::vector<DbTuple>* db = &myDB;



    for (i64 treeIdx = 0; treeIdx < numTrees; ++treeIdx)
    {
        std::vector<DbTuple> randomDb;


        for (u64 i = 0; i < myDB.size(); ++i)
        {
            randomDb.push_back(myDB[prng.get<u64>() % myDB.size()]);
        }
        //std::array<u64, 2> idxs;
        //idxs[0] = prng.get<u64>() % 4;
        //idxs[1] = prng.get<u64>() % 4;

        //while (idxs[0] == idxs[1])
        //{
        //    idxs[1] = prng.get<u64>() % 4;
        //}


        //for (u64 i = 0; i < db->size(); ++i)
        //{
        //    (*db)[i].mPreds = (*db)[i].mPreds2[idxs[0]];
        //    (*db)[i].mPreds.insert((*db)[i].mPreds.end(), (*db)[i].mPreds2[idxs[1]].begin(), (*db)[i].mPreds2[idxs[1]].end());
        //}



        // learn a  simple decision tree
        mTrees[treeIdx].learn(randomDb, minSplit, true);

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



double RandomForest::test(
    std::vector<DbTuple>& testData,
    double learningRate)
{
    //std::vector<DbTuple> updatedTestData(testData);

    auto* evalData = &testData;
    u64 YSq, YSum;

    {

        // compute the predictions for each eval example.
        // Compute the L1, L2, and max error.

        double YSq = 0, YSum = 0;
        double maxY = 0;

        double correct = 0;
        for (u64 i = 0; i < evalData->size(); i++)
        {
            auto y = (*evalData)[i].mValue;
            //auto yprime = mTrees[treeIdx].evaluate((*evalData)[i]) * learningRate;

            auto yprime = evaluate((*evalData)[i]);

            auto Lprime = y - yprime;

            bool cc = std::abs(Lprime) < .5;

            //std::cout << "test[" << i << "]  = " << y << "  pred " << yprime << "  " << cc << std::endl;


            if (cc) ++correct;
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

        double percent = (correct * 100 / evalData->size());
        //std::cout
        //    << " dt " << w << totalDepth << " "
        //    << " #t " << w << mNumTrees << " "
        //    << " l1 " << w << l1
        //    << " l2 " << w << l2
        //    << " max " << w << maxY
        //    << "  " << w << percent << "%" << std::endl;

        return percent;
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

