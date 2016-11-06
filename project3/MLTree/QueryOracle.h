#pragma once
#include <vector>
#include "Common\Defines.h"


typedef double YType;

class DbTuple;

struct splitUpdate
{

	YType mYSum;
	u64 mSize;
};

class QueryOracle
{
public:
	QueryOracle();
	~QueryOracle();


	virtual void init( const YType& ySum, const u64& numRows)  = 0;

	virtual YType setYMax(const YType& y) = 0;
	virtual u64 getNextSplit(const std::vector<std::array<splitUpdate, 2>>& updates, u64 idx, YType splitVal) = 0;
	virtual YType getNextLeafValue(std::vector<YType>&, u64 idx) = 0;
};

