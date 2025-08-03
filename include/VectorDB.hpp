#ifndef VectorDB_Hpp_
#define VectorDB_Hpp_

#include <vector>
#include <string>

enum VectorDBErrorCode {
    VectorDB_OK=0,
    VectorDB_Internal_Error=1,
    VectorDB_Input_Error=2,
};

class VectorDB {
public:
    static VectorDB* createVectorDB(int dim, const std::string& db_file = ":memory:", 
                                    const std::string& distance_metric="cosine",
                                    const std::string& vector_db_type="sqlite-vec");

    // insertion
    virtual VectorDBErrorCode insertVector(int64_t ref, const float* vector) = 0;
    virtual int64_t insertText(const std::string& text) = 0; // return the id of the text
    virtual VectorDBErrorCode insertVectorTextPair(const float* vector, const std::string& text) {
        int64_t ref = insertText(text);
        if (ref < 0) {
            return VectorDB_Internal_Error;
        }
        return insertVector(ref, vector);
    }
    // get
    virtual VectorDBErrorCode getTextbyId(std::vector<int64_t> ids,
                                          std::vector<std::string>& texts) = 0;
    // search
    virtual std::vector<std::pair<int64_t, double>> search(const float* query, int k = 1) = 0;    
    virtual VectorDBErrorCode begin() = 0;
    virtual VectorDBErrorCode commit() = 0;
    virtual std::string getVectorDBFileName() const = 0;
    virtual std::string getTextDBFileName() const = 0;
    virtual int maxTextLen() const { return 2048; }

    virtual ~VectorDB() = default;
};

#endif // VectorDB_Hpp_