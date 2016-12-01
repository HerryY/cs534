#pragma once
#include "MLTree/TreeNode.h"

#include <vector>
#include <functional>
#include <mutex>
#include <list>
#include "Common/PRNG.h"
#include <array>
#include <queue>

//typedef std::array<bool, 3> YType;
typedef double YType;


enum class SplitType
{
    Entropy,
    Random,
    L2,
    L2Laplace,
    Dart
};

inline std::string toString(SplitType t)
{
    if (t == SplitType::Entropy) return "Entropy";
    if (t == SplitType::Random) return "Random";
    if (t == SplitType::L2) return "L2";
    if (t == SplitType::L2Laplace) return "L2Laplace";
    if (t == SplitType::Dart) return "Dart";
    throw std::runtime_error(LOCATION);
}
class MLTree
{
public:
    MLTree();
    ~MLTree();

    MLTree(MLTree&&);


    double mMuteFactor;
    PRNG mPrng;
    std::list<TreeNode*> mLeafNodes;

    struct QueueItem
    {
        QueueItem(TreeNode* node, double loss)
            : mLoss(loss)
            , mNode(node)
        {}

        double mLoss;
        TreeNode* mNode;

        bool operator<(const QueueItem& cmp) const
        {
            return mLoss > cmp.mLoss;
        }

    };
        

    std::priority_queue< QueueItem> nextList;
    u64 mNodeCount;

    TreeNode root;
    u64 mDepth;

    void learn(std::vector<DbTuple>& myDB, u64 mMinSplitSize, u64 maxDepth,
        u64 maxLeafCount, SplitType type, double nodeEpsilon = 1);

    void selectFeatures(std::vector<DbTuple> & db, SplitType type);

    void randomSplit(TreeNode * cur, const u64 &minSplitSize);
    void L2Split(TreeNode * cur, const u64 &minSplitSize);
    void L2LaplaceSplit(TreeNode * cur, const u64 &minSplitSize, double nodeEpsilon);

    void entropySplit(TreeNode * cur, const u64 &predSize, const u64 &minSplitSize);

    YType evaluate(
         const DbTuple& data);

    std::vector<u8> mFeatureSelection;

    u64 getDepth();
    u64 leafCount();
    u64 mLeafCount;
private: 
    //void deleteNode(TreeNode*& node);
};

