#pragma once
#include "MLTree.h"
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
		u64 numTrees,
		double learningRate,
        u64 minSplit,
		std::vector<DbTuple>* evalData = nullptr);


	double test(
		std::vector<DbTuple>& testData,
		double learningRate);

	double evaluate(const DbTuple&);

	double mLearningRate;

	u64 getTotalDepth();
	u64 mNumTrees;
};

