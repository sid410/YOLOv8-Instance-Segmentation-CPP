#ifndef SEGMENTATION_MASK_H
#define SEGMENTATION_MASK_H

#include "OrtApiWrapper/AutoBackendOnnx.h"
#include "IDetectionStrategy.h"
#include <memory>
#include <opencv2/opencv.hpp>

class SegmentationMask
{
private:
    std::unique_ptr<AutoBackendOnnx> model;
    std::unique_ptr<IDetectionStrategy> detectionStrategy;
    void initializeModel(const std::string &modelPath);

public:
    SegmentationMask(const std::string &modelPath, std::unique_ptr<IDetectionStrategy> strategy);
    cv::Mat generateMask(const cv::Mat &img);
    cv::Mat processResults(const cv::Mat &img, const std::vector<YoloResults> &results);
    cv::Mat processResultsDebug(const cv::Mat &img, const cv::Mat &mask);
};

#endif
