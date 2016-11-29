#include "BoostedMLTree.h"

#include "MLTree.h"

#include <fstream>
#include <numeric>
BoostedMLTree::BoostedMLTree()
    :mOut(&std::cout)
{
}


BoostedMLTree::~BoostedMLTree()
{
}

void BoostedMLTree::learn(
    std::vector<DbTuple>& myDB,
    u64 numTrees,
    double learningRate,
    u64 minSplit,
    u64 maxDepth,
    u64 maxLeafCount,
    SplitType type,
    double epsilon,
    std::vector<DbTuple>* evalData)
{
    mLearningRate = learningRate;
    mTrees.reset(new MLTree[numTrees]);

    mNumTrees = numTrees;
    double maxY(9999);

    std::vector<DbTuple>* db = &myDB;
    std::vector<DbTuple> updatedDB(myDB);



    for (i64 treeIdx = 0; treeIdx < numTrees; ++treeIdx)
    {
        // seed the random number generator 
        mTrees[treeIdx].mPrng.SetSeed(mPrng.get<u64>());

        // learn a  simple decision tree
        mTrees[treeIdx].learn(*db, minSplit, maxDepth, maxLeafCount, type, epsilon);


        // now subtract off learningRate * prediction from our labels.
        // This will be come our new dataset.
        for (u64 i = 0; i < myDB.size(); ++i)
        {
            auto Lprime =
                updatedDB[i].mValue -
                learningRate * mTrees[treeIdx].evaluate(updatedDB[i]);

            //if (i < 10)
            //{
            //    std::cout << "db[" << treeIdx << "][" << i << "] : " << updatedDB[i].mValue << " -> " << Lprime << std::endl;
            //}

            updatedDB[i].mValue = Lprime;
        }

        // set the updated data as the training data
        db = &updatedDB;


        // evalute the current performance of the model
        if (evalData)//&& ((treeIdx % 10 == 0) || treeIdx == numTrees - 1)
        {

            // compute the predictions for each eval example.
            // Compute the L1, L2, and max error.

            test(*evalData, std::string("eval ") + std::to_string(treeIdx));

            //double YSq = 0, YSum = 0;
            //maxY = 0;

            //double train_correct = 0;
            //for (u64 i = 0; i < db->size(); i++)
            //{
            //    auto y = (*db)[i].mValue;

            //    auto yprime = evaluate((*db)[i]);

            //    auto Lprime = y - yprime;

            //    if (std::abs(Lprime) < .5) ++train_correct;

            //}

            //double correct = 0;
            //for (u64 i = 0; i < evalData->size(); i++)
            //{
            //    auto y = (*evalData)[i].mValue;
            //    //auto yprime = mTrees[treeIdx].evaluate((*evalData)[i]) * learningRate;

            //    auto yprime = evaluate((*evalData)[i]);

            //    if (i < 10)
            //    {
            //        std::cout << "test[" << treeIdx << "][" << i << "] : " << y << " -> " << yprime << std::endl;
            //    }


            //    auto Lprime = y - yprime;


            //    if (std::abs(Lprime) < .5) ++correct;
            //    //(*evalData)[i].mValue = Lprime;

            //    YSum += std::abs(Lprime);
            //    YSq += Lprime * Lprime;


            //    if (std::abs(Lprime) > maxY)
            //    {
            //        maxY = std::abs(Lprime);
            //    }
            //}


            //auto w = std::setw(8);
            //auto totalDepth = getTotalDepth();
            //double l1 = double(YSum) / (*evalData).size();
            //double l2 = double(YSq) / (*evalData).size();

            //std::cout
            //    << " dt " << w << totalDepth << " "
            //    << " k " << w << minSplit << " "
            //    << " l1 " << w << l1
            //    << " l2 " << w << l2
            //    << " max " << w << maxY
            //    //				<< " train " << w << (train_correct * 100 / db->size()) << "%"
            //    //                << " test  " << w << (correct * 100 / evalData->size()) <<"%"
            //    << std::endl;
        }


        for (auto& leaf : mTrees[treeIdx].mLeafNodes)
        {
            leaf->mRows.clear();
        }
    }
}



double BoostedMLTree::test(
    std::vector<DbTuple>& testData,
    std::string name)
{

    // compute the predictions for each eval example.
    // Compute the L1, L2, and max error.
    auto* evalData = &testData;
    double YSq = 0, YSum = 0;
    double maxY = 0;

    u64 maxIdx = -1;

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
            maxIdx = (*evalData)[i].mIdx;
            maxY = std::abs(Lprime);
        }
    }


    auto w = std::setw(8);
    auto totalDepth = getTotalDepth();
    double l1 = double(YSum) / (*evalData).size();
    double l2 = double(YSq) / (*evalData).size();
    double percent = (correct * 100 / evalData->size());
    *mOut
        << name << " dt " << w << totalDepth << " "
        //<< " k " << w << minSplit << " "
        << " l1 " << w << l1
        << " l2 " << w << l2
        << " max " << w << maxY << "  " << maxIdx
        //<< " train " << w << (train_correct * 100 / db->size()) << "%"
        //<< " test  " << w <<  << "%" 
        << std::endl;

    return percent;
}

double BoostedMLTree::evaluate(const DbTuple & data)
{
    double y = 0;
    for (i64 treeIdx = 0; treeIdx < mNumTrees; ++treeIdx)
    {

        auto yprime = mTrees[treeIdx].evaluate(data);
        auto Lprime = yprime * mLearningRate;


        //std::cout
        //    << "  y' =" << y << " + " << mLearningRate * yprime << std::endl
        //    << "     = " << y + Lprime
        //    //<< "  y' = " << yprime 
        //    << std::endl << std::endl;

        y += Lprime;
        //std::cout<< y << std::endl;

    }
    //std::cout << std::endl;
    return y;

}

u64 BoostedMLTree::getTotalDepth()
{
    u64 sum(0);
    for (u64 i = 0; i < mNumTrees; ++i)
        sum += mTrees[i].getDepth();


    return sum;
}

