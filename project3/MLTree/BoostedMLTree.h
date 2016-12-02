#pragma once
#include "MLTree.h"
#include <vector>
#include <functional>
#include "Common/PRNG.h"
#include <atomic>

class BoostedMLTree
{
public:
	BoostedMLTree();
	~BoostedMLTree();


	std::vector<MLTree> mTrees;

	void learn(
		const std::vector<DbTuple>& myDB,
		u64 numTrees,
		double learningRate,
        u64 minSplit,
        u64 maxDepth,
        u64 maxLeafCount,
        SplitType type,
        double epsilon,
        double dartProb,
        std::vector<DbTuple>* evalData = nullptr);

    void dartUpdate(
        const std::vector<DbTuple> &db, 
        std::vector<DbTuple> &updatedDB,
        u64 size,
        double dropProb);

    void boostUpdate(std::vector<DbTuple> &updatedDB, double learningRate, const i64 &treeIdx);

    std::string mBest;
    double bestL2;

    SplitType mType;
    PRNG mPrng;
	double test(
		std::vector<DbTuple>& testData, 
        std::string name = "");

	double evaluate(const DbTuple&);

	double mLearningRate;

    std::ostream* mOut;


    u64 leafCount();
	u64 getTotalDepth();
	u64 mNumTrees;


    std::atomic<u64>* completedTrees;
};

