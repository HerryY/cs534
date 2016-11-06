#include "DiffPrivQueryOracle.h"



#include <numeric>
struct Score
{
public:
	Score(YType a, YType b, YType c)
	{
		idx = a; gain = b; noisyGain = c;
	}

	Score(const Score& c)
	{
		idx = c.idx;
		noisyGain = c.noisyGain;
		gain = c.gain;
	}
	Score(Score&& c)
	{
		idx = c.idx;
		noisyGain = c.noisyGain;
		gain = c.gain;
	}
	const Score&  operator=(const Score& c)
	{
		idx = c.idx;
		noisyGain = c.noisyGain;
		gain = c.gain;
		return c;
	}
	u64 idx;
	YType noisyGain, gain;
};



DiffPrivQueryOracle::DiffPrivQueryOracle(
	double epsilonPer,
	double epsilonPerLeaf,
	YType maxYValue,
	YType endYValue,
	u64 maxDepth,
	u64 minSplitSize,
	u64 numTrees)
	:
	mNodeLaplace(0, maxYValue * maxYValue / epsilonPer),
	mLeafLaplace(0, maxYValue / epsilonPerLeaf),
	mMaxDepth(maxDepth),
	mEpsilonPerLeaf(epsilonPerLeaf),
	mNodeEpsilon(epsilonPer),
	mMinSplitSize(minSplitSize),
	mYMax(maxYValue),
	mYStep((maxYValue - endYValue) / numTrees),
	mTreeIdx(-1)
{
}

DiffPrivQueryOracle::~DiffPrivQueryOracle()
{

}

void DiffPrivQueryOracle::init(const YType & ySum, const u64 & numRows)
{
	mNodes.clear();

	mNodes.insert({ 1, plainNode() });
	//mLeafs.clear();

	mNodes[1].mYSum += ySum;
	mNodes[1].mNumRows += numRows;
	mNodes[1].mIdx = 1;
	mNodes[1].mDepth = 0;

	mTreeIdx++;
	//mLeafLaplace = Laplace(prng.get<u64>(), mYMax / mEpsilonPerLeaf);
	mEpsilonPerLeaf *= mEpRate;
	mNodeEpsilon *= mEpRate;

}

u64 DiffPrivQueryOracle::getNextSplit(const std::vector<std::array<splitUpdate, 2>>& updates, u64 idx, YType splitVal)
{
#define TRACKING

	auto iter = mNodes.begin();

	{
		std::lock_guard<std::mutex> lock(mNodeMtx);
		iter = mNodes.find(idx);
	}

	if (iter == mNodes.end())
		throw std::runtime_error("");
	auto& node = iter->second;


	u64 bestIdx = -1;
	YType bestNoisyGain = 0, bestGain;
	PRNG prng((idx * mTreeIdx));



	if (node.mDepth < mMaxDepth)
	{


		std::vector<Score> gains;// (updates.size());



		if (0)
		{

			YType maxAvg = 0;
			u64 minI = mMinSplitSize;
			for (u64 i = 0; i < updates.size(); ++i)
			{
				if (updates[i][0].mSize >= mMinSplitSize &&
					updates[i][1].mSize >= mMinSplitSize)
				{
					auto avg0 = std::abs(updates[i][0].mYSum / (YType)updates[i][0].mSize);
					auto avg1 = std::abs(updates[i][1].mYSum / (YType)updates[i][1].mSize);

					maxAvg = std::max(maxAvg, avg0);
					maxAvg = std::max(maxAvg, avg1);

					minI = std::min(minI, std::min(updates[i][0].mSize, updates[i][1].mSize));
				}
			}


			YType scale
				= splitVal * splitVal / minI
				+ 4 * splitVal * maxAvg
				+ maxAvg * maxAvg;

			Laplace lap(prng.get<u64>(), scale / mNodeEpsilon);

			for (u64 i = 0; i < updates.size(); ++i)
			{
				if (updates[i][0].mSize >= mMinSplitSize &&
					updates[i][1].mSize >= mMinSplitSize)
				{

					auto n1 = lap.get();

					YType noisyGain
						= (i64)updates[i][0].mYSum * (i64)updates[i][0].mYSum / (i64)updates[i][0].mSize
						+ (i64)updates[i][1].mYSum * (i64)updates[i][1].mYSum / (i64)updates[i][1].mSize + n1
						;

					YType gain
						= (i64)updates[i][0].mYSum * (i64)updates[i][0].mYSum / (i64)updates[i][0].mSize
						+ (i64)updates[i][1].mYSum * (i64)updates[i][1].mYSum / (i64)updates[i][1].mSize
						;


					//noisyGain = gain;

					gains.emplace_back(i, gain, noisyGain);
					//noisySum += noisyGain;
					//sum += gain;


					if (noisyGain > bestNoisyGain)
					{
						//std::cout<< "lp " << lp2 << " / " << curGain << " =  " << lp2 / curGain << "  ( "<< updates[i][0].mSize << " " << updates[i][1].mSize << " )"<< std::endl;
						bestNoisyGain = noisyGain;
						bestGain = gain;
						bestIdx = i;
					}
				}
			}
		}
		else
		{
			std::vector<bool> sizeChecks(updates.size());
			Laplace lap1(prng.get<u64>(), 1 / mNodeEpsilon);


			for (u64 i = 0; i < updates.size(); ++i)
			{
				sizeChecks[i] =
					updates[i][0].mSize - std::abs(lap1.get()) >= mMinSplitSize &&
					updates[i][1].mSize - std::abs(lap1.get()) >= mMinSplitSize;
			}

			i64 minI = mMinSplitSize;
			YType maxAvg = 0;
			i64 maxDelta = 0;

			for (u64 i = 0; i < updates.size(); ++i)
			{
				if (sizeChecks[i])
				{
					auto avg0 = std::abs(updates[i][0].mYSum / (YType)updates[i][0].mSize);
					auto avg1 = std::abs(updates[i][1].mYSum / (YType)updates[i][1].mSize);

					maxAvg = std::max(maxAvg, std::max(avg0, avg1));
					minI = std::min(minI, std::min((i64)updates[i][0].mSize, (i64)updates[i][1].mSize));
				}
			}

			auto d = splitVal* splitVal / minI;

			for (u64 i = 0; i < updates.size(); ++i)
			{
				if (sizeChecks[i])
				{
					i64 d0 
						= (2 * splitVal * std::abs(updates[i][0].mYSum) + splitVal * splitVal) / updates[i][0].mSize
						+ std::abs(lap1.get()) * d;

					i64 d1
						= (2 * splitVal * std::abs(updates[i][1].mYSum) + splitVal * splitVal) / updates[i][1].mSize
						+ std::abs(lap1.get()) * d;

					maxDelta = std::max(maxDelta, std::max(d0, d1));

				}
			}
			//YType scale
			//	= splitVal * splitVal / minI
			//	+ 4 * splitVal * maxAvg
			//	+ maxAvg * maxAvg;

			//std::cout<< maxDelta << "  " << scale << std::endl;

			for (u64 i = 0; i < updates.size(); ++i)
			{
				if (sizeChecks[i])
				{

					auto n1 = lap1.get();

					YType noisyGain
						= (i64)updates[i][0].mYSum * (i64)updates[i][0].mYSum / (i64)updates[i][0].mSize
						+ (i64)updates[i][1].mYSum * (i64)updates[i][1].mYSum / (i64)updates[i][1].mSize 
						+ n1 * maxDelta
						;

					YType gain
						= (i64)updates[i][0].mYSum * (i64)updates[i][0].mYSum / (i64)updates[i][0].mSize
						+ (i64)updates[i][1].mYSum * (i64)updates[i][1].mYSum / (i64)updates[i][1].mSize
						;


					//noisyGain = gain;

					gains.emplace_back(i, gain, noisyGain);
					//noisySum += noisyGain;
					//sum += gain;


					if (noisyGain > bestNoisyGain)
					{
						//std::cout<< "lp " << lp2 << " / " << curGain << " =  " << lp2 / curGain << "  ( "<< updates[i][0].mSize << " " << updates[i][1].mSize << " )"<< std::endl;
						bestNoisyGain = noisyGain;
						bestGain = gain;
						bestIdx = i;
					}
				}
			}

		}

		
		std::sort(gains.begin(), gains.end(), [](const Score& v1, const Score& v2)
		{
			return v1.gain > v2.gain;
		});
		for (u64 i = 0; i < gains.size(); ++i)
		{
			//std::cout<< gains[i].gain << std::endl;

			if (gains[i].idx == bestIdx)
			{
				//std::cout<< i << " / " << gains.size() <<"   " << gains[i].idx << "  " << idx<< std::endl;
				auto maxSize = std::max(updates[bestIdx][0].mSize, updates[bestIdx][1].mSize);
				auto minSize = std::min(updates[bestIdx][0].mSize, updates[bestIdx][1].mSize);
				auto percentile = 100 * (1 - double(i) / gains.size());
				auto opt = gains[i].gain / gains[0].gain;
				auto relativeNoise = bestNoisyGain / gains[i].gain;;

				//std::cout<< percentile << std::endl;
				//std::cout<< 100 * ( 1 - double(i) / gains.size()) << std::endl;
				mNodePredIdx.emplace_back(percentile, opt, relativeNoise, maxSize / minSize);

				//if ( mTreeIdx > 20)
				//{
				//	std::cout<< gains[0].gain << "  " << gains[0].noisyGain << std::endl;
				//	std::cout<< gains[i].gain << "  " << gains[i].noisyGain << std::endl << std::endl;

				//}
				//break;
			}
		}

	}

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

		node.mLeft->mYSum = updates[bestIdx][0].mYSum;
		node.mLeft->mNumRows = updates[bestIdx][0].mSize;
		node.mLeft->mIdx = node.mIdx << 1;
		node.mLeft->mDepth = node.mDepth + 1;


		node.mRight->mYSum = updates[bestIdx][1].mYSum;
		node.mRight->mNumRows = updates[bestIdx][1].mSize;
		node.mRight->mIdx = node.mIdx << 1 | 1;
		node.mRight->mDepth = node.mDepth + 1;
	}

	return bestIdx;

}
YType DiffPrivQueryOracle::getNextLeafValue(std::vector<YType>& NodeData, u64 idx)
{

	//auto& node = *mLeafs[idx];
	if (1)
	{

		std::sort(NodeData.begin(), NodeData.end());

		//auto startIdx = NodeData.size() * 1 / 20;
		//auto endIdx = NodeData.size() * 19 / 20;


		auto startIdx = 0;
		auto endIdx = NodeData.size();

		YType scale = NodeData[endIdx - 1] - NodeData[startIdx];
		scale = 50;

		YType trimmedSum = 0;
		for (auto i = startIdx; i < endIdx; ++i)
		{
			trimmedSum += NodeData[i];
		}



		{
			//auto sum = std::accumulate(NodeData.begin(), NodeData.end(), 0.0);
			//if (std::abs(sum - node.mYSum) > 0.0001)
			//	throw std::runtime_error("");

			//if(node.mNumRows !=NodeData.size())
			//	throw std::runtime_error("");

			//if(scale > 100)
			//	throw std::runtime_error("");

		}

		// round the outliers into the range.
		//trimmedSum += startIdx * (NodeData[startIdx] + NodeData[endIdx - 1]);
		PRNG prng((idx * mTreeIdx));

		Laplace lap(prng.get<u64>(), scale / mEpsilonPerLeaf);
		auto ret = i64(lap.get() + trimmedSum) * 1024 / (i64)(endIdx - startIdx);//NodeData.size();//





		return  YType(ret) / 1024;
	}
	else if (0)
	{


		YType trimmedSum = 0;
		for (auto i = 0; i < NodeData.size(); ++i)
		{
			trimmedSum += NodeData[i];
		}


		auto ret = (trimmedSum) / (YType)(NodeData.size());//NodeData.size();//



		return ret;
	}
	else
	{

		return mMedian.getMedian(NodeData, mYMax, mEpsilonPerLeaf);


	}
}

YType DiffPrivQueryOracle::setYMax(const YType & maxYValue)
{
	//mYMax -= mYStep;  //maxYValue;
	//mYMax = std::max(0.0, mYMax - 1);
	PRNG prng(((u64)maxYValue));

	//std::cout<< "setting maxY " << mYMax << std::endl;
	//mLeafLaplace = Laplace(prng.get<u64>(), mYMax / mEpsilonPerLeaf);
	//mNodeLaplace = Laplace(prng.get<u64>(), mYMax * mYMax / mNodeEpsilon);


	return mYMax;
}

void DiffPrivQueryOracle::setEpRate(double rate)
{
	mEpRate = rate;

}
