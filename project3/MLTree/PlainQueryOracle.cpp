#include "PlainQueryOracle.h"

#include "TreeNode.h"

PlainQueryOracle::PlainQueryOracle(u64 maxDepth, u64 minSplitSize)
	:mMaxDepth(maxDepth),
	mMinSplitSize(minSplitSize)
{
}


PlainQueryOracle::~PlainQueryOracle()
{
}

void PlainQueryOracle::init(const YType & ySum, const u64 & numRows)
{

	mNodes.clear();

	mNodes.insert({ 1, plainNode() });

	mNodes[1].mIdx = 1;
	mNodes[1].mDepth = 0;


}

u64 PlainQueryOracle::getNextSplit(const std::vector<std::array<splitUpdate, 2>>& updates, u64 idx, YType splitVal)
{



	auto iter = mNodes.begin();

	{
		std::lock_guard<std::mutex> lock(mNodeMtx);
		iter = mNodes.find(idx);

		if (iter == mNodes.end())
			throw std::runtime_error("");
	}
	auto& node = iter->second;



	//auto& node = mFirstCall ? *(mCurNode = mNodes.begin()) : *++mCurNode;

	u64 bestIdx = -1;

	i64 bestGain = 99999999999999;


	if (node.mDepth < mMaxDepth)
	{
		for (u64 i = 0; i < updates.size(); ++i)
		{
			if (updates[i][0].mSize > mMinSplitSize &&
				updates[i][1].mSize > mMinSplitSize)
			{


				i64 gain
					= -(i64)updates[i][0].mYSum * (i64)updates[i][0].mYSum / (i64)updates[i][0].mSize
					+ -(i64)updates[i][1].mYSum * (i64)updates[i][1].mYSum / (i64)updates[i][1].mSize
					;

				//std::cout
				//	<< gain << " = " << (i64)updates[i][0].mYSum << " * " << (i64)updates[i][0].mYSum << " / " << (i64)updates[i][0].mSize << std::endl
				//	<< "       + " << (i64)updates[i][1].mYSum << " * " << (i64)updates[i][1].mYSum << " / " << (i64)updates[i][1].mSize << std::endl;

				if (gain < bestGain)
				{
					bestGain = gain;
					bestIdx = i;
				}
			}
		}
	}
	//std::cout<< idx << "  " << bestIdx << std::endl;

	if (bestIdx == -1)
	{
	}
	else
	{

		{
			std::lock_guard<std::mutex> lock(mNodeMtx);
			mNodes.insert({ node.mIdx << 1 , plainNode() });
			mNodes.insert({ node.mIdx << 1 | 1, plainNode() });
		}
		node.mLeft = &mNodes[node.mIdx << 1];
		node.mRight = &mNodes[node.mIdx << 1 | 1];

		node.mLeft->mIdx = node.mIdx << 1;
		node.mLeft->mDepth = node.mDepth + 1;

		node.mRight->mIdx = node.mIdx << 1 | 1;
		node.mRight->mDepth = node.mDepth + 1;

	}
	return bestIdx;
}


#include <algorithm>
#include <numeric>
YType PlainQueryOracle::getNextLeafValue(std::vector<YType>& data, u64 idx)
{
	i64  sum = std::accumulate(data.begin(), data.end(), i64(0));
	auto ret = sum / (i64)data.size();
	return  ret;
}

YType PlainQueryOracle::setYMax(const YType & y)
{
	return 100;
}

