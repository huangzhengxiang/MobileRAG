#ifndef MNNRAG_Hpp_
#define MNNRAG_Hpp_

#include "RAG.hpp"
#include "llm/llm.hpp"
#include <memory>

using namespace MNN;
using namespace MNN::Transformer;

class MNNRAG : public RAG {
public:
    MNNRAG();
    virtual RAGErrorCode loadDB(int   dim, 
                                const std::string& vector_db_path = ":memory:",
                                bool  from_disk=false,
                                const std::string& distance_metric="cosine",
                                int   top_k=2) override;
    virtual RAGErrorCode loadEmbedding(const std::string& config_path) override;  
    virtual RAGErrorCode loadGenerator(const std::string& config_path) override;
    virtual RAGErrorCode insertDB(const std::vector<std::string>& docs) override; // currently only supports text
    virtual RAGErrorCode saveDB() override;
    virtual std::string generate(const std::string& question, const std::vector<std::string>& docs) override;
    virtual std::string query(const std::string& question) override;
    ~MNNRAG();

private:
    int embed_dim;
    int db_top_k;
    std::unique_ptr<VectorDB> db;
    std::unique_ptr<Embedding> embedding;
    std::unique_ptr<Llm> llm;
};

#endif // MNNRAG_Hpp_