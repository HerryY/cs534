#pragma once
#include "MLTree/TreeNode.h"
#include "MLTree/QueryOracle.h"

#include <vector>
#include <functional>
#include <mutex>
#include <list>

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

	void learn(
		 std::vector<DbTuple>& myDB, 
		QueryOracle& qo);

	void test(
		 std::vector<DbTuple>& testData);

	YType evaluate(
		 const DbTuple& data);


	u64 getDepth();

private: 
	void deleteNode(TreeNode*& node);
};

