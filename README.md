# MoPiRAG

MoPiRAG: Fast and Efficient Mobile Pipeline-Parallelized RAG with Multi-Level Hot and Cold Caching

## O. Installation

1. install sqlite-vec extension
[sqlite](https://alexgarcia.xyz/sqlite-vec/compiling.html)
```bash
git clone https://github.com/asg017/sqlite-vec
cd sqlite-vec
./scripts/vendor.sh
make loadable
```

2. build
- build for linux
```bash
mkdir -p build/pc && cd build/pc
cmake ../../ -DCMAKE_BUILD_TYPE=Release
make -j16
```

- build for android cli
```bash
export ANDROID_NDK=<path-to-android-ndk>
mkdir -p build/phone && cd build/phone
cmake ../../ -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="arm64-v8a" -DANDROID_STL=c++_static -DANDROID_NATIVE_API_LEVEL=android-28 -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=.
make -j16
```