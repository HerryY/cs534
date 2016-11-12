#pragma once
#include "MLTree.h"
#include <vector>
#include <functional>



class RandomForest
{
public:
	RandomForest();
	~RandomForest();


	std::unique_ptr<MLTree[]> mTrees;

	void learn(
		std::vector<DbTuple>& myDB,
		u64 numTrees,
        u64 minSplit,
		std::vector<DbTuple>* evalData = nullptr);


	void test(
		std::vector<DbTuple>& testData,
		double learningRate);

	double evaluate(const DbTuple&);


	u64 getTotalDepth();
	u64 mNumTrees;
};

