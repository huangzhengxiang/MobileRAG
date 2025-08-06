#include "RAG.hpp"
#include "MNNRAG.hpp"

RAG* RAG::createRAG(RAGTypeCode type,
                    std::ostream* log) {
    if (type==RAGType_MNNRAG) {
        return new MNNRAG(log);
    } else {
        printf("RAG type unsupported!\n");
        return nullptr;
    }
}