#pragma once
#include "MLTree/TreeNode.h"

#include <vector>
#include <functional>
#include <mutex>
#include <list>
#include "Common/PRNG.h"
#include <array>

//typedef std::array<bool, 3> YType;
typedef double YType;

enum class SplitType
{
    Entropy,
    Random,
    L2
};

class MLTree
{
public:
    MLTree();
    ~MLTree();

    PRNG mPrng;
    std::mutex mNextListMtx, mLeafListMtx;
    std::list<TreeNode*> nextList, mLeafNodes;
    u64 mNodeCount;


    TreeNode root;
    u64 mDepth;

    void learn(std::vector<DbTuple>& myDB, u64 mMinSplitSize, SplitType type);

    void selectFeatures(std::vector<DbTuple> & db, SplitType type);

    void randomSplit(TreeNode * cur, const u64 &minSplitSize);
    void L2Split(TreeNode * cur, const u64 &minSplitSize);

    void entropySplit(TreeNode * cur, const u64 &predSize, const u64 &minSplitSize);

    YType evaluate(
         const DbTuple& data);

    std::vector<u8> mFeatureSelection;

    u64 getDepth();

private: 
    void deleteNode(TreeNode*& node);
};

