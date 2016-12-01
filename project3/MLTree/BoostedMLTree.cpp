#include "BoostedMLTree.h"

#include "MLTree.h"

#include <fstream>
#include <numeric>
BoostedMLTree::BoostedMLTree()
    :mOut(&std::cout),
    bestL2(9999999999999)
{
}


BoostedMLTree::~BoostedMLTree()
{
}

void BoostedMLTree::learn(
    const std::vector<DbTuple>& myDB,
    u64 numTrees,
    double learningRate,
    u64 minSplit,
    u64 maxDepth,
    u64 maxLeafCount,
    SplitType type,
    double epsilon,
    double dartProb,
    std::vector<DbTuple>* evalData)
{
    mLearningRate = learningRate;
    mTrees.resize(numTrees);
    mNumTrees = 0;
    double maxY(9999);

    std::vector<DbTuple> updatedDB(myDB);
    mType = type;

    for (i64 treeIdx = 0; treeIdx < numTrees; ++treeIdx)
    {
        // seed the random number generator 
        mTrees[treeIdx].mPrng.SetSeed(mPrng.get<u64>());

        // learn a  simple decision tree
        mTrees[treeIdx].learn(updatedDB, minSplit, maxDepth, maxLeafCount, type, epsilon);
        ++mNumTrees;


        if (evalData)
        {
            test(*evalData, std::string("@") + std::to_string(treeIdx));
        }


        if (type != SplitType::Random && type != SplitType::Dart)
        {
            boostUpdate(updatedDB, learningRate, treeIdx);
        }
        else if (type == SplitType::Dart && treeIdx != numTrees - 1)
        {
            dartUpdate(myDB, updatedDB,  treeIdx + 1, dartProb);
        }


        for (auto& leaf : mTrees[treeIdx].mLeafNodes)
        {
            leaf->mRows.clear();
        }
    }
}

void BoostedMLTree::dartUpdate(
    const std::vector<DbTuple> &db,
    std::vector<DbTuple> &updatedDB,
    u64 size, 
    double dropProb)
{
    std::vector<u8> dropList(size);
    u64 k = 0;
    for (u64 i = 0; i < dropList.size(); ++i)
    {
        double v = mPrng.get<u32>();

        double threshold = (u32(-1) * dropProb);

        dropList[i] = (v < threshold);

        if (dropList[i]) ++k;

        //std::cout << " dart " << i << " " << v << " > " << threshold << "  " << dartProb << std::endl;
    }
    if (k == 0)
    {
        dropList[mPrng.get<u32>() % dropList.size()] = 1;
        ++k;
    }


    // now subtract off learningRate * prediction from our labels.
    // This will be come our new dataset.
    for (u64 i = 0; i < updatedDB.size(); ++i)
    {
        double sum = 0;
        for (u64 j = 0; j < dropList.size(); ++j)
        {
            if (dropList[j] == 0)
            {
                sum +=  mTrees[j].evaluate(updatedDB[i]) * mTrees[j].mMuteFactor;
            }
        }

        updatedDB[i].mValue = db[i].mValue - sum;
    }

    for (u64 i = 0; i < dropList.size(); ++i)
    {
        if (dropList[i])
        {
            mTrees[i].mMuteFactor *= k / (1.0 + k);
        }
    }

    mTrees[dropList.size()].mMuteFactor = 1.0 / (1.0 + k);

}

void BoostedMLTree::boostUpdate(std::vector<DbTuple> &updatedDB, double learningRate, const i64 &treeIdx)
{
    // now subtract off learningRate * prediction from our labels.
    // This will be come our new dataset.
    for (u64 i = 0; i < updatedDB.size(); ++i)
    {
        auto Lprime =
            updatedDB[i].mValue -
            learningRate * mTrees[treeIdx].evaluate(updatedDB[i]);

        updatedDB[i].mValue = Lprime;
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

    auto lc = leafCount();
    std::stringstream ss;
    ss
        << name << " dt " << w << totalDepth << " "
        //<< " k " << w << minSplit << " "
        << " lc " << w << lc
        << " l1 " << w << l1
        << " l2 " << w << l2
        << " max " << w << maxY << "  " << maxIdx
        //<< " train " << w << (train_correct * 100 / db->size()) << "%"
        //<< " test  " << w <<  << "%" 
        << std::endl;

    *mOut << ss.str() << std::flush;

    if (bestL2 > l2)
    {
        bestL2 = l2;

        mBest = ss.str();
    }

    return percent;
}

double BoostedMLTree::evaluate(const DbTuple & data)
{
    double y = 0;

    if (mType == SplitType::Random)
    {
        for (i64 treeIdx = 0; treeIdx < mNumTrees; ++treeIdx)
        {
            y += mTrees[treeIdx].evaluate(data);
        }

        y = y / mNumTrees;
    }
    else if (mType == SplitType::Dart)
    {
        for (i64 treeIdx = 0; treeIdx < mNumTrees; ++treeIdx)
        {
            y += mTrees[treeIdx].evaluate(data) * mTrees[treeIdx].mMuteFactor;
        }
    }
    else
    {

        for (i64 treeIdx = 0; treeIdx < mNumTrees; ++treeIdx)
        {

            y += mTrees[treeIdx].evaluate(data) * mLearningRate;
        }
    }

    //std::cout << std::endl;
    return y;

}
//
//std::vector<u8> BoostedMLTree::sampleDropList(u64 size, double dropProb)
//{
//
//}

u64 BoostedMLTree::leafCount()
{
    u64 sum(0);
    for (u64 i = 0; i < mNumTrees; ++i)
        sum += mTrees[i].leafCount();


    return sum;
}

u64 BoostedMLTree::getTotalDepth()
{
    u64 sum(0);
    for (u64 i = 0; i < mNumTrees; ++i)
        sum += mTrees[i].getDepth();


    return sum;
}

