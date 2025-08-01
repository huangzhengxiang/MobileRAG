#include <vector>
#include <string>

enum VectorDBErrorCode {
    VectorDB_OK=0,
    VectorDB_Internal_Error=1,
    VectorDB_Input_Error=2,
};

class VectorDB {
public:
    VectorDB* createVectorDB(int dim, const std::string& db_file = ":memory:", 
                             const std::string& distance_metric="cosine");

    virtual VectorDBErrorCode insert(int64_t ref, const float* vector) = 0;
    virtual std::vector<std::pair<int64_t, double>> search(const float* query, int k = 1) = 0;    
    virtual VectorDBErrorCode begin() = 0;
    virtual VectorDBErrorCode commit() = 0;

    virtual ~VectorDB() = default;
};