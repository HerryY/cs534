#include "PRNG.h"
#include <algorithm>
#include <cstring>




#define DEFAULT_BUFF_SIZE 64
    PRNG::PRNG() : mBytesIdx(0), mBlockIdx(0),
        mBuffer(DEFAULT_BUFF_SIZE),
        mIndexArray(DEFAULT_BUFF_SIZE, 0),
        mBufferByteCapacity(sizeof(u64) * DEFAULT_BUFF_SIZE)
    {
    }


    PRNG::PRNG(const u64& seed)
        : 
        mBytesIdx(0), 
        mBlockIdx(0),
        mBuffer(DEFAULT_BUFF_SIZE),
        mIndexArray(DEFAULT_BUFF_SIZE, 0),
        mBufferByteCapacity(sizeof(u64) * DEFAULT_BUFF_SIZE)
    {
        mSeed = seed;
        rng.seed(seed);
         


        refillBuffer();
    }

    PRNG::PRNG(PRNG && s) :
        mSeed(s.mSeed),
        mBuffer(std::move(s.mBuffer)),
        mIndexArray(std::move(s.mIndexArray)),
        rng(std::move(s.rng)),
        mBytesIdx(s.mBytesIdx),
        mBlockIdx(s.mBlockIdx),
        mBufferByteCapacity(s.mBufferByteCapacity)
    {
        s.mSeed = 0;
        s.mBuffer.resize(0);
        s.mIndexArray.resize(0);
        s.mBytesIdx = 0;
        s.mBlockIdx = 0;
        s.mBufferByteCapacity = 0;
    }


    void PRNG::SetSeed(const u64& seed)
    {
        mSeed = seed;
        rng.seed(seed);
        mBlockIdx = 0;

        if (mBuffer.size() == 0)
        {
            mBuffer.resize(DEFAULT_BUFF_SIZE);
            mIndexArray.resize(DEFAULT_BUFF_SIZE);
            mBufferByteCapacity = (sizeof(u64) * DEFAULT_BUFF_SIZE);
        }


        refillBuffer();
    }

    const u64 PRNG::getSeed() const
    {
        return mSeed;
    }

    void PRNG::refillBuffer()
    {
        for (u64 i = 0; i < mBuffer.size(); ++i)
        {
            mBuffer[i] = dist6(rng);
        }

        mBytesIdx = 0;
    }
