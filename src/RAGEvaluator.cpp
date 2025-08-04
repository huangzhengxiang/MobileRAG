#include "RAGEvaluator.hpp"
#include <string>
#include <vector>

bool checkContainCorrects(const std::string& ans, 
                          const std::vector<std::string>& ans_list) {
    for (const auto& a : ans_list) {
        if (ans.find(a) != std::string::npos) {
            return true;
        }
    }
    return false;
}