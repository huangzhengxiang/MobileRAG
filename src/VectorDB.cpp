#include "VectorDB.hpp"
#include "SQLiteVectorDB.hpp"

VectorDB* VectorDB::createVectorDB(int dim, const std::string& db_file, 
                                          const std::string& distance_metric,
                                          const std::string& vector_db_type) {
    if (vector_db_type=="sqlite-vec") {
        return new SQLiteVectorDB(dim, db_file, distance_metric);
    } else {
        printf("VectorDB type unsupported!\n");
        return nullptr;
    }
}