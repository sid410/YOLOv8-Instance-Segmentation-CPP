#include "SegmentationMask.h"
#include <opencv2/opencv.hpp>
#include <iostream>

SegmentationMask::SegmentationMask(const std::string &modelPath, std::unique_ptr<IDetectionStrategy> strategy)
    : detectionStrategy(std::move(strategy))
{
    initializeModel(modelPath);
}

void SegmentationMask::initializeModel(const std::string &modelPath)
{
    const char *onnx_provider = OnnxProviders::CPU.c_str(); // Ensure OnnxProviders::CPU is correctly defined
    const char *onnx_logid = "segmentation";
    model = std::make_unique<AutoBackendOnnx>(modelPath.c_str(), onnx_logid, onnx_provider);
    // Assume AutoBackendOnnx has a method to initialize and load the model
}

cv::Mat SegmentationMask::generateMask(const cv::Mat &img)
{
    cv::Mat converted_img;
    cv::cvtColor(img, converted_img, cv::COLOR_BGR2RGB);

    // Create variables for the floating-point arguments
    float conf_threshold = 0.30f;
    float iou_threshold = 0.45f;
    float mask_threshold = 0.5f;
    int conversion_code = cv::COLOR_BGR2RGB;

    // Pass variables instead of literals to predict_once
    auto results = model->predict_once(converted_img, conf_threshold, iou_threshold, mask_threshold, conversion_code);

    auto filteredResults = detectionStrategy->filterResults(results);
    return processResults(img, filteredResults);
}

cv::Mat SegmentationMask::processResults(const cv::Mat &img, const std::vector<YoloResults> &results)
{
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());

    for (const auto &result : results)
    {
        if (result.mask.rows > 0 && result.mask.cols > 0)
        {
            mask(result.bbox).setTo(cv::Scalar(255, 255, 255), result.mask);
        }
    }

    return mask;
}

cv::Mat SegmentationMask::processResultsDebug(const cv::Mat &img, const cv::Mat &mask)
{
    cv::Mat highlightedImg = img.clone();
    cv::Mat colorMask(img.size(), CV_8UC3, cv::Scalar(0, 255, 0)); // Green mask
    colorMask.copyTo(highlightedImg, mask);
    cv::addWeighted(highlightedImg, 0.5, img, 0.5, 0, highlightedImg, img.type());
    return highlightedImg;
}
