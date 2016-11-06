#pragma once
#include "QueryOracle.h"
#include <list>
#include <array>
#include "Laplace.h"
#include "DiffPrivMedian.h"
#include <unordered_map>
#include <mutex>

struct TreeStats
{
	TreeStats(double x, double y, double z, double d)
	{
		averageSplitIdx = (x);
		averageSplitOpt = (y);
		relativeNoise = z;
		relativeSize = d;
	}

	double averageSplitIdx;
	double averageSplitOpt;
	double relativeNoise, relativeSize;
};

class DiffPrivQueryOracle :
	public QueryOracle
{
public:
	DiffPrivQueryOracle(
		double epsilonPer, 
		double epsilonPerLeaf, 
		YType maxYValue, 
		YType endYValue,
		u64 maxDepth, 
		u64 minSplitSize,
		u64 numTrees);

	~DiffPrivQueryOracle();


	void init(const YType& ySum, const u64& numRows) override;
	u64 getNextSplit(const std::vector<std::array<splitUpdate, 2>>& updates, u64 idx, YType splitVal) override;
	YType getNextLeafValue(std::vector<YType>&, u64 idx) override;
	YType setYMax(const YType& y) override;

	void setEpRate(double rate);
	struct plainNode
	{
		u64 mNumRows, mIdx, mDepth;
		YType mYSum;


		plainNode * mLeft, *mRight;
	};

	DiffPrivMedian mMedian;
	double mEpsilonPerLeaf, mNodeEpsilon, mEpRate;
	Laplace mNodeLaplace, mLeafLaplace;
	u64 mMaxDepth, mMinSplitSize, mTreeIdx;
	YType mYMax, mYStep;

	std::vector<TreeStats> mNodePredIdx;

	std::mutex mNodeMtx;
	std::unordered_map<u64, plainNode> mNodes;
	//std::unordered_map<u64, plainNode*> mLeafs;
	//std::list<plainNode>::iterator mCurNode;
	//std::list<plainNode*>::iterator mNextLeaf;
};

