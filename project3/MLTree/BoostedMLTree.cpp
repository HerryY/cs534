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
    mTrees.reset(new MLTree[numTrees]);
    mNumTrees = 0;
    mNumTrees = numTrees;
    double maxY(9999);

    std::vector<DbTuple> updatedDB(myDB);

    mType = type;

    for (i64 treeIdx = 0; treeIdx < numTrees; ++treeIdx)
    {
        // seed the random number generator 
        mTrees[treeIdx].mPrng.SetSeed(mPrng.get<u64>());

        // learn a  simple decision tree
        mTrees[treeIdx].learn(updatedDB, minSplit, maxDepth, maxLeafCount, type, epsilon);
        //++mNumTrees;

        if (type != SplitType::Random && type != SplitType::Dart)
        {
            boostUpdate(updatedDB, learningRate, treeIdx);
        }
        else if (type == SplitType::Dart)
        {
            dartUpdate(treeIdx, dartProb, myDB, updatedDB, learningRate);

        }

        // evalute the current performance of the model
        if (evalData)//&& ((treeIdx % 10 == 0) || treeIdx == numTrees - 1)
        {

            // compute the predictions for each eval example.
            // Compute the L1, L2, and max error.

            test(*evalData, std::string("@") + std::to_string(treeIdx));
        }


        for (auto& leaf : mTrees[treeIdx].mLeafNodes)
        {
            leaf->mRows.clear();
        }
    }
}

void BoostedMLTree::dartUpdate(
    const i64 &treeIdx, 
    double dartProb, 
    const std::vector<DbTuple> &db,
    std::vector<DbTuple> &updatedDB,
    double learningRate)
{
    std::vector<u8> dropList(treeIdx + 1);
    for (u64 i = 0; i <= treeIdx; ++i)
    {
        double v = mPrng.get<u32>();

        double threshold = (u32(-1) * dartProb);

        dropList[i] = (v < threshold);


        //std::cout << " dart " << i << " " << v << " > " << threshold << "  " << dartProb << std::endl;
    }

    // now subtract off learningRate * prediction from our labels.
    // This will be come our new dataset.
    for (u64 i = 0; i < updatedDB.size(); ++i)
    {
        double sum = 0;
        for (u64 j = 0; j <= treeIdx; ++j)
        {
            if (dropList[j] == 0)
            {
                TreeNode* cur = &mTrees[j].root;

                // traverse the tree until we get to a leaf. 
                while (cur->mPredIdx[0] != -1)
                {

                    auto px = updatedDB[i].mPredsGroup[cur->mPredIdx[0]][cur->mPredIdx[1]];
                    cur = px ? cur->mRight.get() : cur->mLeft.get();

                }

                sum += cur->mValue;
            }
        }

        auto Lprime =
            db[i].mValue -
            learningRate * sum;

        //if (i < 10)
        //{
        //    std::cout << "db[" << treeIdx << "][" << i << "] : " << updatedDB[i].mValue << " -> " << Lprime << std::endl;
        //}

        updatedDB[i].mValue = Lprime;
    }

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

        //if (i < 10)
        //{
        //    std::cout << "db[" << treeIdx << "][" << i << "] : " << updatedDB[i].mValue << " -> " << Lprime << std::endl;
        //}

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

