#pragma once
#include "MLTree.h"
#include "QueryOracle.h"
#include <vector>
#include <functional>



class BoostedMLTree
{
public:
	BoostedMLTree();
	~BoostedMLTree();


	std::unique_ptr<MLTree[]> mTrees;

	void learn(
		std::vector<DbTuple>& myDB,
		QueryOracle& qo,
		u64 numTrees,
		double learningRate,
		std::vector<DbTuple>* evalData = nullptr);


	void test(
		std::vector<DbTuple>& testData,
		double learningRate);

	YType evaluate(const DbTuple&);

	double mLearningRate;

	u64 getTotalDepth();
	u64 mNumTrees;
};

