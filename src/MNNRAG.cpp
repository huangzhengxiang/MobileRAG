#include "MNNRAG.hpp"
#include <iostream>
#include <sstream>
#include <string>

using namespace MNN::Express;

MNNRAG::MNNRAG() {

}

RAGErrorCode MNNRAG::loadDB(int   dim, 
                            const std::string& vector_db_path,
                            bool  from_disk,
                            const std::string& distance_metric,
                            int   top_k) {
    embed_dim = dim;
    db_top_k = top_k;
    db.reset(VectorDB::createVectorDB(embed_dim, vector_db_path, distance_metric)); 
    return RAG_OK;
}


RAGErrorCode MNNRAG::loadEmbedding(const std::string& config_path) {
    std::cout << "Embedding config path is " << config_path << std::endl;
    embedding.reset(Embedding::createEmbedding(config_path, true));
    if (embedding.get()!=nullptr) { return RAG_OK; }
    else { return Embedding_Error; }
}                          

RAGErrorCode MNNRAG::loadGenerator(const std::string& config_path) {
    std::cout << "Llm config path is " << config_path << std::endl;
    llm.reset(Llm::createLLM(config_path));
    if (llm.get()==nullptr) { return Generator_Error; }
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    {
        llm->load();
    }
    if (true) {
        llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    }
    return RAG_OK;    
}                    

RAGErrorCode MNNRAG::insertDB(const std::vector<std::string>& docs) {
    db->begin();
    for (const auto& doc: docs) {
        auto varp = embedding->txt_embedding(doc, embed_dim);
        auto code = db->insertVectorTextPair(varp->readMap<float>(), doc);
        if (code != VectorDB_OK){
            return VectorDB_Error;
        }
    }
    db->commit();
    return RAG_OK;
}

RAGErrorCode MNNRAG::saveDB() {
    return RAG_OK; 
}


std::string MNNRAG::generate(const std::string& question, const std::vector<std::string>& docs) {
    std::string prompt = question + "\nRelated documents are:";
    for (const auto& doc : docs) {
        prompt += doc+"\n";
    }
    std::ostringstream oss;
    llm->response(prompt, &oss, "\n");
    return oss.str();
}

std::string MNNRAG::query(const std::string& question) {
    // auto start = time();
    auto query = embedding->txt_embedding(question, embed_dim);
    // coarse-grained search
    auto id_dists = db->search(query->readMap<float>(), db_top_k);
    // auto end = time();
    // printf("embed+retrieval: %.2fs\n", end-start); 
    std::vector<int64_t> ids;
    for (auto id_dist : id_dists) {
        // printf("rowid=%lld distance=%f\n", id_dist.first, id_dist.second);
        ids.push_back(id_dist.first);
    }
    std::vector<std::string> docs;
    db->getTextbyId(ids, docs);
    return generate(question, docs);
}

MNNRAG::~MNNRAG() {}