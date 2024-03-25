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
    // common
    static std::vector<std::string> parseVectorString(const std::string &input);
    static std::vector<int> convertStringVectorToInts(const std::vector<std::string> &input);
    static std::unordered_map<int, std::string> parseNames(const std::string &input);
    static int64_t vector_product(const std::vector<int64_t> &vec);

    // augment
    static void letterbox(const cv::Mat &image,
                          cv::Mat &outImage,
                          const cv::Size &newShape = cv::Size(640, 640),
                          cv::Scalar_<double> color = cv::Scalar(), bool auto_ = true,
                          bool scaleFill = false,
                          bool scaleUp = true,
                          int stride = 32);
    static void scaleImage(
        cv::Mat &scaled_mask, const cv::Mat &resized_mask, const cv::Size &im0_shape,
        const std::pair<float, cv::Point2f> &ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)));

    // ops
    cv::Rect_<float> static scale_boxes(const cv::Size &img1_shape, cv::Rect_<float> &box, const cv::Size &img0_shape, std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), bool padding = true);

    static void clip_boxes(cv::Rect &box, const cv::Size &shape);
    static void clip_boxes(cv::Rect_<float> &box, const cv::Size &shape);
    static void clip_boxes(std::vector<cv::Rect> &boxes, const cv::Size &shape);
    static void clip_boxes(std::vector<cv::Rect_<float>> &boxes, const cv::Size &shape);
};

#endif