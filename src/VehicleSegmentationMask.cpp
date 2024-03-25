// VehicleSegmentationMask.cpp
#include "VehicleSegmentationMask.h"
#include <iostream>
#include <algorithm>

VehicleSegmentationMask::VehicleSegmentationMask(const std::string &modelPath)
{
    initializeModel(modelPath);
}

void VehicleSegmentationMask::initializeModel(const std::string &modelPath)
{
    const char *onnx_provider = OnnxProviders::CPU.c_str(); // Assume OnnxProviders::CPU is a std::string constant or variable
    const char *onnx_logid = "vehicle_segmentation";
    // Assuming AutoBackendOnnx constructor requires const char* arguments
    model = std::make_unique<AutoBackendOnnx>(modelPath.c_str(), onnx_logid, onnx_provider);
    names = model->getNames();
}

cv::Mat VehicleSegmentationMask::generateMask(const cv::Mat &img)
{
    cv::Mat converted_img;
    cv::cvtColor(img, converted_img, conversion_code);
    std::vector<YoloResults> results = model->predict_once(converted_img, conf_threshold, iou_threshold, mask_threshold, conversion_code);

    // Filter results to include only the allowed classes
    std::vector<YoloResults> filteredResults;
    std::copy_if(results.begin(), results.end(), std::back_inserter(filteredResults),
                 [this](const YoloResults &result)
                 {
                     return allowedClasses.find(result.class_idx) != allowedClasses.end();
                 });

    return processResults(img, filteredResults); // Use filtered results here
}

cv::Mat VehicleSegmentationMask::processResults(const cv::Mat &img, const std::vector<YoloResults> &results)
{
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type()); // Initialize mask as black

    for (const auto &result : results)
    {
        // Check for valid mask and apply white color to detected objects
        if (result.mask.rows > 0 && result.mask.cols > 0)
        {
            mask(result.bbox).setTo(cv::Scalar(255, 255, 255), result.mask);
        }
    }

    return mask; // Return the final mask
}

cv::Mat VehicleSegmentationMask::processResultsDebug(const cv::Mat &img, const cv::Mat &mask)
{
    cv::Mat highlightedImg = img.clone();
    // Assume the mask is binary (0 or 255). If it's not, you might need to threshold it.

    // Create a color mask to overlay
    cv::Mat colorMask(img.size(), CV_8UC3, cv::Scalar(0, 255, 0)); // Green mask
    colorMask.copyTo(highlightedImg, mask);

    // Blend the original image with the color mask where the mask is white
    cv::addWeighted(highlightedImg, 0.5, img, 0.5, 0, highlightedImg, img.type());

    return highlightedImg;
}