#pragma once
#include "QueryOracle.h"
#include <list>
#include <array>
#include <unordered_map>
#include <mutex>

class PlainQueryOracle :
	public QueryOracle
{
public:
	PlainQueryOracle(u64 maxDepth, u64 minSplitSize);
	~PlainQueryOracle();

	void init(const YType& ySum, const u64& numRows) override;
	u64 getNextSplit(const std::vector<std::array<splitUpdate, 2>>& updates, u64 idx, YType splitVal);// override;
	YType getNextLeafValue(std::vector<YType>&, u64 idx) override;
	YType setYMax(const YType& y) override;

	struct plainNode
	{
		u64 mIdx, mDepth;
		//YType mYSum;
		plainNode * mLeft, *mRight;
	};

	u64 mMaxDepth, mMinSplitSize;


	std::mutex mNodeMtx;
	std::unordered_map<u64, plainNode> mNodes;

	
};

