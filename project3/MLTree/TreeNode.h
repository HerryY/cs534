#pragma once

#include <vector>
#include "Common\Defines.h"
#include <istream>

#include <ostream>
#include <array>
class DbTuple
{

public:
    std::array<std::vector<u8>, 4> mPreds2;
   
    std::vector<u8> mPreds;
	std::vector<double> mPlain;
	double mValue;
};

class TreeNode
{
public:
	TreeNode();
	~TreeNode();

	TreeNode * mRight, *mLeft;
	u64 mPredIdx, mIdx, mDepth;
	double mValue;

	std::vector<DbTuple*> mRows;

    double hash();

	void toFile(std::ostream& out);
};

