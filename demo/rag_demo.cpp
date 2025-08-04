//
//  vectordb_demo.cpp
//
//  Created by MNN on 2024/01/10.
//  ZhaodeWang
//

#include "MNNRAG.hpp"
#include "dataset.hpp"
#include "RAGEvaluator.hpp"
#include "llm/llm.hpp"
#include <fstream>
#include <stdlib.h>

using namespace MNN::Express;
using namespace MNN::Transformer;

#define DUMP_NUM_DATA(type)                          \
    auto data = var->readMap<type>();                \
    for (int z = 0; z < outside; ++z) {              \
        for (int x = 0; x < width; ++x) {            \
            outputOs << data[x + z * width] << "\n"; \
        }                                            \
    }

static void dumpVar2File(VARP var, const char* file) {
    std::ofstream outputOs(file);

    auto dimension = var->getInfo()->dim.size();
    int width     = 1;
    if (dimension > 1) {
        width = var->getInfo()->dim[dimension - 1];
    }

    auto outside = var->getInfo()->size / width;
    DUMP_NUM_DATA(float);

}

static void dumpVARP(VARP var) {
    auto size = static_cast<int>(var->getInfo()->size);
    auto ptr = var->readMap<float>();
    printf("[ ");
    for (int i = 0; i < 5; i++) {
        printf("%f, ", ptr[i]);
    }
    printf("... ");
    for (int i = size - 5; i < size; i++) {
        printf("%f, ", ptr[i]);
    }
    printf(" ]\n");
}

static void unittest(std::unique_ptr<Embedding> &embedding, std::string prompt) {
    auto vec_0 = embedding->txt_embedding(prompt);
    float sum = 0;
    auto ptr = vec_0->readMap<float>();
    for (int i = 0;i < vec_0->getInfo()->size; ++i) {
        sum += ptr[i];
    }
    MNN_PRINT("%s\n", prompt.c_str());
    MNN_PRINT("sum = %f\n", sum);
    MNN_PRINT("\n");
}
static void benchmark_trivia_qa(std::unique_ptr<MNNRAG>& rag, 
                                int build_db,
                                const std::string& dataset_file) {
    std::string data;
    std::vector<std::string> docs;
    std::vector<std::string> questions;
    std::vector<std::vector<std::string>> answers;

    // Read file contents into 'data'
    std::ifstream in_file(dataset_file);
    if (!in_file) {
        std::cout << "Failed to open dataset file: " << dataset_file << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << in_file.rdbuf();
    data = buffer.str();
    
    // read in dataset
    process_trivia_qa(data, docs, 
                      questions, answers);
    
    // build vectorDB
    if (build_db) {
        printf("build vector DB!\n");
        rag->insertDB(docs);
    }

    int acc = 0;
    // evaluate
    for (int i=0; i<questions.size(); ++i) {
        auto response = rag->query("Please answer the question： "+questions[i]+"\nPlease output the answer only!\n");
        std::cout << response << std::endl;
        if (checkContainCorrects(response, answers[i])) {
            acc++;
        }
    }
    printf("accuracy: %.2f%%\n", (acc*100.0f)/questions.size());
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " embedding_config.json llm_config.json [build_db] [dataset.json]" << std::endl;
        return 0;
    }
    std::string embedding_config_path = argv[1];
    std::string llm_config_path = argv[2];

    std::unique_ptr<MNNRAG> rag(new MNNRAG());
    rag->loadDB(1024, "./test.db");
    rag->loadEmbedding(embedding_config_path);
    rag->loadGenerator(llm_config_path);

    if (argc >= 4) {
        benchmark_trivia_qa(rag, std::stoi(argv[3]), argv[4]);
        return 0;
    }

    std::vector<std::string> docs = {"在春暖花开的季节，走在樱花缤纷的道路上，人们纷纷拿出手机拍照留念。樱花树下，情侣手牵手享受着这绝美的春光。孩子们在树下追逐嬉戏，脸上洋溢着纯真的笑容。春天的气息在空气中弥漫，一切都显得那么生机勃勃，充满希望。",
        "春天到了，樱花树悄然绽放，吸引了众多游客前来观赏。小朋友们在花瓣飘落的树下玩耍，而恋人们则在这浪漫的景色中尽情享受二人世界。每个人的脸上都挂着幸福的笑容，仿佛整个世界都被春天温暖的阳光和满树的樱花渲染得更加美好。",
        "在炎热的夏日里，沙滩上的游客们穿着泳装享受着海水的清凉。孩子们在海边堆沙堡，大人们则在太阳伞下品尝冷饮，享受悠闲的时光。远处，冲浪者们挑战着波涛，体验着与海浪争斗的刺激。夏天的海滩，总是充满了活力和热情。"};
    rag->insertDB(docs);
    // search+generation
    auto response = rag->query("概述下春天");
    printf("query: %s, response: %s\n", "概述下春天", response.c_str());
    return 0;
}
