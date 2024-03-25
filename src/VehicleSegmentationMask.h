#ifndef VEHICLE_SEGMENTATION_MASK_H
#define VEHICLE_SEGMENTATION_MASK_H

#include "OrtApiWrapper/AutoBackendOnnx.h"
#include "OrtApiWrapper/OnnxModelBase.h"
#include "OrtApiWrapper/YoloUtils.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <unordered_set>

class VehicleSegmentationMask
{
public:
    VehicleSegmentationMask(const std::string &modelPath);
    cv::Mat generateMask(const cv::Mat &img);
    cv::Mat processResultsDebug(const cv::Mat &img, const cv::Mat &mask);

private:
    std::unique_ptr<AutoBackendOnnx> model;
    std::unordered_map<int, std::string> names;
    std::unordered_set<int> allowedClasses = {1, 2, 3, 5, 6, 7}; // The COCO class indices for Vehicle

    float mask_threshold = 0.5f;
    float conf_threshold = 0.30f;
    float iou_threshold = 0.45f;
    int conversion_code = cv::COLOR_BGR2RGB;

    void initializeModel(const std::string &modelPath);
    cv::Mat processResults(const cv::Mat &img, const std::vector<YoloResults> &results);
};

#endif