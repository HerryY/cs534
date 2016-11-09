#pragma once
#include "MLTree/TreeNode.h"

#include <vector>
#include <functional>
#include <mutex>
#include <list>

//typedef std::array<bool, 3> YType;
typedef double YType;

struct splitUpdate
{

    YType mYSum;
    u64 mSize;
};


class MLTree
{
public:
    MLTree();
    ~MLTree();

    std::mutex mNextListMtx, mLeafListMtx;
    std::list<TreeNode*> nextList, mLeafNodes;
    u64 mNodeCount;


    TreeNode root;
    u64 mDepth;

    void learn(std::vector<DbTuple>& myDB, u64 mMinSplitSize);

    YType evaluate(
         const DbTuple& data);


    u64 getDepth();

private: 
    void deleteNode(TreeNode*& node);
};

