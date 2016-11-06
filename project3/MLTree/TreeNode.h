#pragma once

#include <vector>
#include "Common\Defines.h"
#include "Common\BitVector.h"
#include <istream>

#include "QueryOracle.h"
#include <ostream>

class DbTuple
{

public:
	//BitVector mPreds;
	std::vector<u8> mPreds;
	std::vector<double> mPlain;
	YType mValue;
	/*i64 getValue() const;
	void setValue(i64 newVal);*/
};

class TreeNode
{
public:
	TreeNode();
	~TreeNode();

	TreeNode * mRight, *mLeft;
	u64 mPredIdx, mIdx, mDepth;
	YType mValue;

	std::vector<DbTuple*> mRows;

    YType hash();

	void toFile(std::ostream& out);
	//void loadFromFile(std::istream& in, double scale);
};

