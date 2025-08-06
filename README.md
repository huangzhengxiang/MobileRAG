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

run android cli
```bash
# under build/phone
mkdir -p /data/local/tmp/llm
adb push librag.so libsqlite-vec.so rag_demo MNN/arm64-v8a/libMNN.so MNN/arm64-v8a/libllm.so MNN/express/arm64-v8a/libMNN_Express.so MNN/tools/audio/arm64-v8a/libMNNAudio.so MNN/tools/cv/arm64-v8a/libMNNOpenCV.so dataset/libdataset.so /data/local/tmp/llm/
```

adb shell
```bash
./rag_demo model/Qwen3-Embedding-0.6B-MNN/config.json model/qwen2_5-1_5b-instruct-int4-mnn/config.json 0 ./val00-100.json
```