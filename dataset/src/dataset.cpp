#include "dataset.hpp"

void parse_json_list_str(const rapidjson::Value& array, 
                         std::vector<std::string>& strings) {
    for (auto& item : array.GetArray()) {
        if (item.IsString()) {
            strings.push_back(item.GetString());
        } else {
            printf("array element not string!\n");
        }
    }
}

// dialog, turn, 
void process_trivia_qa(std::string& data, 
                       std::vector<std::string>& docs,
                       std::vector<std::string>& questions,
                       std::vector<std::vector<std::string>>& answers) {
    rapidjson::Document document;
    document.Parse(data.c_str());

    // Process "docs"
    if (document.HasMember("docs")) {
        const auto& val = document["docs"];
        if (val.IsString()) {
            docs.push_back(val.GetString());
        } else if (val.IsArray()) {
            parse_json_list_str(val, docs);
        } else {
            printf("docs format unsupported!\n");
        }
    }

    // Process "questions"
    if (document.HasMember("questions")) {
        const auto& val = document["questions"];
        if (val.IsString()) {
            questions.push_back(val.GetString());
        } else if (val.IsArray()) {
            parse_json_list_str(val, questions);
        } else {
            printf("questions format unsupported!\n");
        }
    }

    // Process "answers"
    if (document.HasMember("answers")) {
        const auto& val = document["answers"];
        if (val.IsArray()) {
            for (const auto& item : val.GetArray()) {
                if (item.IsArray()) {
                    std::vector<std::string> answer_set;
                    parse_json_list_str(item, answer_set);
                    answers.push_back(std::move(answer_set));
                } else {
                    printf("answers format unsupported!\n");
                }
            }
        }
    }
}