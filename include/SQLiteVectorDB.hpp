#include <VectorDB.hpp>
#include "sqlite3.h"
#include "sqlite-vec.h"
#include <vector>
#include <string>

class SQLiteVectorDB : public VectorDB {
public:
    SQLiteVectorDB(int dim, const std::string& db_file = ":memory:", 
                   const std::string& distance_metric="cosine");
    virtual VectorDBErrorCode insert(int64_t ref, const float* vector) override;
    virtual std::vector<std::pair<int64_t, double>> search(const float* query, int k = 1) override;
    virtual VectorDBErrorCode begin() override;
    virtual VectorDBErrorCode commit() override;  
    virtual ~SQLiteVectorDB() override;

protected:
    VectorDBErrorCode errorCode(int rc) {
        return (rc == SQLITE_DONE) ? VectorDB_OK : VectorDB_Internal_Error;
    }
    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    std::string insert_sql = "INSERT INTO vec_items(ref, embedding) VALUES (?, ?)";
    std::string search_sql = "SELECT ref, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT ?";
    int embed_dim;
};