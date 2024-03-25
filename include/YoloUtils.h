#ifndef YOLO_UTILS_H
#define YOLO_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <regex>

class YoloUtils
{
public:
    static std::vector<std::string> parseVectorString(const std::string &input);
    static std::vector<int> convertStringVectorToInts(const std::vector<std::string> &input);
    static std::unordered_map<int, std::string> parseNames(const std::string &input);
    static int64_t vector_product(const std::vector<int64_t> &vec);
};

#endif