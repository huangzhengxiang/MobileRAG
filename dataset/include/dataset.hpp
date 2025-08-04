#ifndef DATASET_Hpp_
#define DATASET_Hpp_


#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <iterator>
#include <random>
#include "dataset.hpp"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "rapidjson/error/en.h"

#include <iostream>
#include <map>
#include <string>

void parse_json_list_str(const rapidjson::Value& array, 
                         std::vector<std::string>& strings);

void process_trivia_qa(std::string& data, 
                       std::vector<std::string>& docs,
                       std::vector<std::string>& questions,
                       std::vector<std::vector<std::string>>& answers);

#endif // DATASET_Hpp_