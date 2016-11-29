#pragma once

#include <vector>
#include "Common/Defines.h"
#include <istream>

#include <ostream>
#include <array>
class DbTuple
{

public:
    std::vector<std::vector<u8>> mPredsGroup;
   
    //std::vector<u8> mPreds;
	std::vector<double> mPlain;
	double mValue;
    u64 mIdx;
};

class TreeNode
{
public:
	TreeNode();
	~TreeNode();

	std::unique_ptr<TreeNode> mRight, mLeft;
	u64  mIdx, mDepth;
    std::array<u64, 2> mPredIdx;

	double mValue;

	std::vector<DbTuple*> mRows;

    double hash();

	void toFile(std::ostream& out);
};

