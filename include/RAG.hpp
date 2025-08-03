#ifndef RAG_Hpp_
#define RAG_Hpp_

#include "VectorDB.hpp"
#include <string>
#include <vector>

enum RAGErrorCode {
    RAG_OK=0,
    VectorDB_Error=1,
    Embedding_Error=2,
    Reranker_Error=3,
    Generator_Error=4,
};

enum RAGTypeCode {
    RAGType_MNNRAG=0,
};

class RAG {
public:
    static RAG* createRAG(RAGTypeCode type=RAGType_MNNRAG);
    virtual RAGErrorCode loadDB(int   dim, 
                                const std::string& vector_db_path = ":memory:",
                                bool  from_disk=false,
                                const std::string& distance_metric="cosine",
                                int   top_k=2) = 0;
    virtual RAGErrorCode loadEmbedding(const std::string& config_path) = 0;  
    virtual RAGErrorCode loadReranker(const std::string& config_path) {
        return RAG_OK; // default no reranker
    }            
    virtual RAGErrorCode loadGenerator(const std::string& config_path) = 0;
    virtual RAGErrorCode insertDB(const std::vector<std::string>& docs) = 0; // currently only supports text
    virtual RAGErrorCode saveDB() = 0;
    virtual std::string generate(const std::string& question, const std::vector<std::string>& docs) = 0;
    virtual std::string query(const std::string& question) = 0;
};

#endif // RAG_Hpp_