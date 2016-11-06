#pragma once

#include "TreeNode.h"
#include "Crypto/PRNG.h"

class DiffPrivMedian
{
public:
	DiffPrivMedian();
	~DiffPrivMedian();

	u64 getIterations(YType maxValue);
	YType getMedian(std::vector<YType>& myDB, YType maxValue, double epsilon);


	double mScaleFrac;
private:
	void getNextRange(YType&, YType&, bool b);
	PRNG mPrng;

};

