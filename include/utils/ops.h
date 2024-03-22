#ifndef OPS_H
#define OPS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

cv::Rect_<float> scale_boxes(const cv::Size &img1_shape, cv::Rect_<float> &box, const cv::Size &img0_shape, std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), bool padding = true);

std::vector<float> scale_coords(const cv::Size &img1_shape, std::vector<float> &coords, const cv::Size &img0_shape);

cv::Mat crop_mask(const cv::Mat &mask, const cv::Rect &box);

void clip_boxes(cv::Rect &box, const cv::Size &shape);
void clip_boxes(cv::Rect_<float> &box, const cv::Size &shape);
void clip_boxes(std::vector<cv::Rect> &boxes, const cv::Size &shape);
void clip_boxes(std::vector<cv::Rect_<float>> &boxes, const cv::Size &shape);

void clip_coords(std::vector<float> &coords, const cv::Size &shape);

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
non_max_suppression(const cv::Mat &output0, int class_names_num, int total_features_num, double conf_threshold, float iou_threshold);

#endif
