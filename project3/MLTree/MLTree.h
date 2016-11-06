#pragma once
#include "MLTree/TreeNode.h"

#include <vector>
#include <functional>
#include <mutex>
#include <list>

struct splitUpdate
{

    double mYSum;
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

    u64 mMaxDepth, mMinSplitSize;

	TreeNode root;
	u64 mDepth;

	void learn(std::vector<DbTuple>& myDB, u64 mMaxDepth, u64 mMinSplitSize);

    u64 getNextSplit(
        const std::vector<std::array<splitUpdate, 2>>& updates,
        TreeNode* cur);

	void test(
		 std::vector<DbTuple>& testData);

	double evaluate(
		 const DbTuple& data);


	u64 getDepth();

private: 
	void deleteNode(TreeNode*& node);
};

