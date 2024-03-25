#ifndef PERSON_SEGMENTATION_MASK_H
#define PERSON_SEGMENTATION_MASK_H

#include "OrtApiWrapper/AutoBackendOnnx.h"
#include "OrtApiWrapper/YoloUtils.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

class PersonSegmentationMask
{
public:
    PersonSegmentationMask(const std::string &modelPath);
    cv::Mat generateMask(const cv::Mat &img);
    cv::Mat processResultsDebug(const cv::Mat &img, const cv::Mat &mask);

private:
    std::unique_ptr<AutoBackendOnnx> model;
    std::unordered_map<int, std::string> names;
    float mask_threshold = 0.5f;
    float conf_threshold = 0.30f;
    float iou_threshold = 0.45f;
    int conversion_code = cv::COLOR_BGR2RGB;

    void initializeModel(const std::string &modelPath);
    cv::Mat processResults(const cv::Mat &img, const std::vector<YoloResults> &results);
};

#endif
