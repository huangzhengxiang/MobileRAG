#ifndef SQLiteVectorDB_Hpp_
#define SQLiteVectorDB_Hpp_

#include <VectorDB.hpp>
#include "sqlite3.h"
#include "sqlite-vec.h"
#include <vector>
#include <string>
#include <memory>

class SQLiteVectorDB : public VectorDB {
public:
    SQLiteVectorDB(int dim, const std::string& db_file = ":memory:", 
                   const std::string& distance_metric="cosine");
    // insertion
    virtual VectorDBErrorCode insertVector(int64_t ref, const float* vector) override;
    virtual int64_t insertText(const std::string& text) override; // return the id of the text
    // get
    virtual VectorDBErrorCode getTextbyId(std::vector<int64_t> ids,
                                          std::vector<std::string>& texts) override;
    // search
    virtual std::vector<std::pair<int64_t, double>> search(const float* query, int k = 1) override;
    virtual VectorDBErrorCode begin() override;
    virtual VectorDBErrorCode commit() override;  
    virtual std::string getVectorDBFileName() const override;
    virtual std::string getTextDBFileName() const override;
    virtual ~SQLiteVectorDB() override;

protected:
    VectorDBErrorCode errorCode(int rc) {
        return (rc == SQLITE_DONE) ? VectorDB_OK : VectorDB_Internal_Error;
    }
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    std::string insert_vector_sql = "INSERT INTO vec_items(ref, embedding) VALUES (?, ?)";
    std::string insert_text_sql = "INSERT INTO text_items(id, text) VALUES (NULL, ?)";
    std::string search_sql = "SELECT ref, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT ?";
    int embed_dim;
};

#endif // SQLiteVectorDB_Hpp_