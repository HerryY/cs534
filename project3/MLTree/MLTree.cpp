#include "MLTree.h"
#include "Common\Defines.h"

#include <array>
#include "Common/Timer.h"
#include <atomic>

MLTree::MLTree()
    :mDepth(0)
{
}


MLTree::~MLTree()
{

    deleteNode(root.mLeft);
    deleteNode(root.mRight);

}



void MLTree::learn(std::vector<DbTuple>& db, u64 depth, u64 split)
{
    mMaxDepth = depth;
    mMinSplitSize = split;

    mDepth = 0;
    mNodeCount = 1;

    root.mIdx = 1;
    root.mDepth = 0;

    root.mRows.resize(db.size());
    for (u64 i = 0; i < db.size(); ++i)
    {
        auto& y = db[i].mValue;

        root.mRows[i] = &db[i];
    }


    u64 predSize = db[0].mPreds.size();

    nextList.push_back(&root);


    std::vector<std::array<splitUpdate, 2>>
        updates(predSize);


    bool thisThreadOutOfWork = false;


    double splitSize(2);
    TreeNode* cur = nullptr;

    while (nextList.size())
    {
        cur = nextList.back();
        nextList.pop_back();

        Timer t;
        t.setTimePoint("nodeStart " + std::to_string((u16)this));

        for (auto& u : updates)
        {
            u[0].mSize = 0;
            u[0].mYSum = 0;

            u[1].mSize = 0;
            u[1].mYSum = 0;
        }


        for (auto j = 0; j < cur->mRows.size(); ++j)
        {
            auto& row = cur->mRows[j];

            auto y = row->mValue;

            for (u64 i = 0; i < predSize; ++i)
            {
                u8 px = row->mPreds[i];

                updates[i][px].mYSum += y;
                updates[i][px].mSize++;

            }
        }

        // split the current node with preds[i]
        cur->mPredIdx = getNextSplit(updates, cur);


        //std::cout<< Log::lock << cur->mIdx << "\t" << (i64)cur->mPredIdx << std::endl << Log::unlock;


        if (cur->mPredIdx != u64(-1))
        {

            std::array<TreeNode*, 2> nodes = { new TreeNode(),new TreeNode() };

            cur->mLeft = nodes[0];
            cur->mRight = nodes[1];

            nodes[0]->mIdx = cur->mIdx << 1;
            nodes[1]->mIdx = cur->mIdx << 1 | 1;
            nodes[0]->mDepth = nodes[1]->mDepth = cur->mDepth + 1;


            for (auto& row : cur->mRows)
            {
                auto& y = row->mValue;

                u8 px = row->mPreds[cur->mPredIdx];

                nodes[px]->mRows.emplace_back(std::move(row));
            }

            cur->mRows.clear();
            cur->mRows.shrink_to_fit();


            nextList.push_back(nodes[0]);
            nextList.push_back(nodes[1]);

            mDepth = std::max(mDepth, nodes[0]->mDepth);

            mNodeCount += 2;

            //std::cout<< Log::lock << mNodeCount << std::endl << Log::unlock;
        }
        else
        {

            mLeafNodes.push_back(cur);
            // this node should not be split any more...
        }
        t.setTimePoint("nodeDone " + std::to_string((u16)this));
    }


    std::vector<double> leafVals;

    while (mLeafNodes.size())
    {
        TreeNode* cur = mLeafNodes.back();
        mLeafNodes.pop_back();

        double  sum = 0;

        leafVals.resize(cur->mRows.size());
        for (auto i = 0; i < cur->mRows.size(); ++i)
        {
            sum  += cur->mRows[i]->mValue;
        }

        cur->mValue = sum / cur->mRows.size();
    }
}


u64 MLTree::getNextSplit(
    const std::vector<std::array<splitUpdate, 2>>& updates,
    TreeNode* mNode)
{


    u64 bestIdx = -1;

    i64 bestGain = 99999999999999;


    if (mNode->mDepth < mMaxDepth)
    {
        for (u64 i = 0; i < updates.size(); ++i)
        {
            if (updates[i][0].mSize > mMinSplitSize &&
                updates[i][1].mSize > mMinSplitSize)
            {


                i64 gain
                    = -(i64)updates[i][0].mYSum * (i64)updates[i][0].mYSum / (i64)updates[i][0].mSize
                    + -(i64)updates[i][1].mYSum * (i64)updates[i][1].mYSum / (i64)updates[i][1].mSize
                    ;

                if (gain < bestGain)
                {
                    bestGain = gain;
                    bestIdx = i;
                }
            }
        }
    }

    return bestIdx;
}


void MLTree::test(
    std::vector<DbTuple>& testData)
{


    u64 averageL1Error(0);
    u64 ySq(0);
    u64 ySum(0);

    for (auto& row : testData)
    {
        TreeNode* cur = &root;

        while (cur->mPredIdx != -1)
        {

            auto px = row.mPreds[cur->mPredIdx];
            cur = px ? cur->mRight : cur->mLeft;

        }

        cur->mRows.push_back(&row);

        averageL1Error += std::abs((i64)cur->mValue - row.mValue);
        //ySq += row.getValue() * row.getValue();
        //ySum += 
    }

    double totalError(0);

    for (auto cur : mLeafNodes)
    {
        double finalYSq = 0, finalYSum = 0;

        for (auto& row : cur->mRows)
        {
            auto& y = row->mValue;

            finalYSq += y * y;
            finalYSum += y;

        }

        if (cur->mRows.size())
        {

            auto error = (finalYSq - finalYSum*finalYSum / cur->mRows.size());
            totalError += error;

            //std::cout<< 
            //	"leaf Idx=" << cur->mIdx << 
            //	"   value = " << cur->mValue << 
            //	"   error = " << error << 
            //	"   out of " << cur->mRows.size() << std::endl;
        }
    }


    std::cout
        << "avg l1 error " << double(averageL1Error) / testData.size() << std::endl
        << "avg l2 error " << totalError / testData.size() << std::endl;

}





double MLTree::evaluate(const DbTuple & row)
{
    TreeNode* cur = &root;

    while (cur->mPredIdx != -1)
    {

        auto px = row.mPreds[cur->mPredIdx];
        cur = px ? cur->mRight : cur->mLeft;

    }

    return cur->mValue;
}

u64 MLTree::getDepth()
{
    return mDepth;
}

void MLTree::deleteNode(TreeNode *& node)
{
    if (node)
    {
        deleteNode(node->mLeft);
        deleteNode(node->mRight);


        delete node;
        node = nullptr;
    }

}
