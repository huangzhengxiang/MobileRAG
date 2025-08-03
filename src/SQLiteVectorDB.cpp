#include "SQLiteVectorDB.hpp"
#include <vector>
#include <iostream>

SQLiteVectorDB::SQLiteVectorDB(int dim, const std::string& db_file,
                               const std::string& distance_metric) {
    embed_dim = dim; // no more than 8192
    if (sqlite3_auto_extension((void(*)())sqlite3_vec_init) != SQLITE_OK) {
        printf("unable to load sqlite-vec extension!\n");
        return;
    }
    if (sqlite3_open(db_file.c_str(), &db) != SQLITE_OK) {
        printf("open db file failure!\n"); // TODO: change to a portable print
        return;
    }
    std::string sql = "CREATE VIRTUAL TABLE IF NOT EXISTS " 
        "vec_items USING vec0(ref integer primary key, embedding float[" + std::to_string(dim) \ 
        + "] distance_metric=" + distance_metric + ")"; 
    // std::cout << sql << std::endl;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK) {
        printf("initialize db failure!\n");
        return;
    }
    sql = "CREATE TABLE IF NOT EXISTS " 
        "text_items (id integer primary key, text TEXT NOT NULL CHECK (length(text) <= 2048))"; 
    // std::cout << sql << std::endl;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK) {
        printf("initialize db failure!\n");
        return;
    }
}

SQLiteVectorDB::~SQLiteVectorDB() {
    if (db) sqlite3_close(db);
}

VectorDBErrorCode SQLiteVectorDB::begin() {
    int rc = sqlite3_exec(db, "BEGIN", NULL, NULL, NULL);
    return errorCode(rc);
}
VectorDBErrorCode SQLiteVectorDB::commit() {
    sqlite3_finalize(stmt); // Invoking sqlite3_finalize() on a NULL pointer is a harmless no-op. 
    int rc = sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
    return errorCode(rc);
}

VectorDBErrorCode SQLiteVectorDB::insertVector(int64_t ref, const float* vector) {
    if (sqlite3_prepare_v2(db, insert_vector_sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) 
        return VectorDB_Internal_Error;

    sqlite3_bind_int64(stmt, 1, ref);
    sqlite3_bind_blob(stmt, 2, vector, sizeof(float) * embed_dim, SQLITE_STATIC);

    int rc = sqlite3_step(stmt);
    sqlite3_reset(stmt);
    return errorCode(rc);
}

int64_t SQLiteVectorDB::insertText(const std::string& text) {
    if (sqlite3_prepare_v2(db, insert_text_sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) 
        return VectorDB_Internal_Error;

    sqlite3_bind_text(stmt, 1, text.c_str(), -1, SQLITE_STATIC);

    int rc = sqlite3_step(stmt);
    if (errorCode(rc) != VectorDB_OK) {
        sqlite3_reset(stmt);
        return -1;
    }

    int64_t row_id = sqlite3_last_insert_rowid(db);
    sqlite3_reset(stmt);
    return row_id;
}

VectorDBErrorCode SQLiteVectorDB::getTextbyId(std::vector<int64_t> ids,
                                              std::vector<std::string>& texts) {
    if (ids.empty()) return VectorDB_OK;

    // Build SQL with IN (?, ?, ..., ?)
    std::string sql = "SELECT text FROM text_items WHERE id IN (";
    for (size_t i = 0; i < ids.size(); ++i) {
        sql += (i == 0 ? "?" : ",?");
    }
    sql += ")";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return VectorDB_Internal_Error;
    }

    // Bind ID values
    for (size_t i = 0; i < ids.size(); ++i) {
        sqlite3_bind_int64(stmt, static_cast<int>(i + 1), ids[i]);
    }

    // Step through results
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char* text = sqlite3_column_text(stmt, 0);
        if (text) {
            texts.emplace_back(reinterpret_cast<const char*>(text));
        } else {
            texts.emplace_back("");  // handle NULLs as empty string
        }
    }

    sqlite3_finalize(stmt);
    return VectorDB_OK;
}

std::vector<std::pair<int64_t, double>> SQLiteVectorDB::search(const float* query, int k) {
    std::vector<std::pair<int64_t, double>> results;
    if (sqlite3_prepare_v2(db, search_sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) return results;

    sqlite3_bind_blob(stmt, 1, query, sizeof(float) * embed_dim, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, k);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t rowid = sqlite3_column_int64(stmt, 0);
        double dist = sqlite3_column_double(stmt, 1);
        results.emplace_back(rowid, dist);
    }

    sqlite3_finalize(stmt);
    return results;
}

std::string SQLiteVectorDB::getVectorDBFileName() const {
    return "vec_items";
}

std::string SQLiteVectorDB::getTextDBFileName() const {
    return "text_items";
}