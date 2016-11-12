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

struct splitUpdate
{

    YType mYSum;
    u64 mSize;
	std::array<u32, 3> classFreq;
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

    void learn(std::vector<DbTuple>& myDB, u64 mMinSplitSize, bool random);

    YType evaluate(
         const DbTuple& data);


    u64 getDepth();

private: 
    void deleteNode(TreeNode*& node);
};

