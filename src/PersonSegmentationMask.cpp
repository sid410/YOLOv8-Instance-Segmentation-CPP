#include "PersonSegmentationMask.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

PersonSegmentationMask::PersonSegmentationMask(const std::string &modelPath)
{
    initializeModel(modelPath);
}

void PersonSegmentationMask::initializeModel(const std::string &modelPath)
{
    const char *onnx_provider = OnnxProviders::CPU.c_str();
    const char *onnx_logid = "person_segmentation";
    model = std::make_unique<AutoBackendOnnx>(modelPath.c_str(), onnx_logid, onnx_provider);
    names = model->getNames();
}

cv::Mat PersonSegmentationMask::generateMask(const cv::Mat &img)
{
    cv::Mat converted_img;
    cv::cvtColor(img, converted_img, conversion_code);
    std::vector<YoloResults> results = model->predict_once(converted_img, conf_threshold, iou_threshold, mask_threshold, conversion_code);

    // Filter results to include only the 'person' class (COCO class index 0)
    std::vector<YoloResults> filteredResults;
    std::copy_if(results.begin(), results.end(), std::back_inserter(filteredResults),
                 [](const YoloResults &result)
                 {
                     return result.class_idx == 0; // Filter for 'person' class
                 });

    std::cout << "Detected People: " << filteredResults.size() << std::endl; // Print the count of detected people

    return processResults(img, filteredResults);
}

cv::Mat PersonSegmentationMask::processResults(const cv::Mat &img, const std::vector<YoloResults> &results)
{
    // Implementation similar to VehicleSegmentationMask, but specific to people
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());

    for (const auto &result : results)
    {
        if (result.mask.rows > 0 && result.mask.cols > 0)
        {
            mask(result.bbox).setTo(cv::Scalar(255, 255, 255), result.mask);
        }
    }

    return mask; // Return the final mask
}

cv::Mat PersonSegmentationMask::processResultsDebug(const cv::Mat &img, const cv::Mat &mask)
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
