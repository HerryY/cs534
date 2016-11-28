#pragma once
#include "MLTree.h"
#include <vector>
#include <functional>
#include "Common/PRNG.h"


class BoostedMLTree
{
public:
	BoostedMLTree();
	~BoostedMLTree();


	std::unique_ptr<MLTree[]> mTrees;

	void learn(
		std::vector<DbTuple>& myDB,
		u64 numTrees,
		double learningRate,
        u64 minSplit,
        u64 maxDepth,
        u64 maxLeafCount,
        SplitType type,
        double epsilon,
        std::vector<DbTuple>* evalData = nullptr);

    PRNG mPrng;
	double test(
		std::vector<DbTuple>& testData, 
        std::string name = "");

	double evaluate(const DbTuple&);

	double mLearningRate;

	u64 getTotalDepth();
	u64 mNumTrees;
};

