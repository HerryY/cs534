#pragma once
#include "Common/Defines.h"
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#define SEED_SIZE   AES_BLK_SIZE
#define RAND_SIZE   AES_BLK_SIZE




    class PRNG
    {
    public:

        u64 mSeed;
        std::vector<u64> mBuffer, mIndexArray;
        u64 mBytesIdx, mBlockIdx, mBufferByteCapacity;
        void refillBuffer();
        std::mt19937 rng;
        std::uniform_int_distribution<std::mt19937::result_type> dist6;



        PRNG();
        PRNG(const u64& seed);
        PRNG(const PRNG&) = delete;
        PRNG(PRNG&& s);


        // Set seed from array
        void SetSeed(const u64& b);
        const u64 getSeed() const;

         
        template<typename T>
        T get()
        {
            static_assert(std::is_pod<T>::value, "T must be POD");
            T ret;
            get((u8*)&ret, sizeof(T));
            return ret;
        }

        template<typename T>
        void get(T* dest, u64 length)
        {
            static_assert(std::is_pod<T>::value, "T must be POD");
            u64 lengthu8 = length * sizeof(T);
            u8* destu8 = (u8*)dest;
            while (lengthu8)
            {
                u64 step = std::min(lengthu8, mBufferByteCapacity - mBytesIdx);

                memcpy(destu8, ((u8*)mBuffer.data()) + mBytesIdx, step);

                destu8 += step;
                lengthu8 -= step;
                mBytesIdx += step;

                if (mBytesIdx == mBufferByteCapacity)
                    refillBuffer();
            }
        }


        u8 getBit() { return get<u8>() & 1; }
        //void get(u8* ans, u64 len);




        typedef u32 result_type;
        static result_type min() { return 0; }
        static result_type max() { return (result_type)-1; }
        result_type operator()() {
            return get<result_type>();
        }
        result_type operator()(int mod) {
            return get<result_type>() % mod;
        }
    };
