// #include "utils/ops.h"

// void clip_boxes(cv::Rect &box, const cv::Size &shape)
// {
//     box.x = std::max(0, std::min(box.x, shape.width));
//     box.y = std::max(0, std::min(box.y, shape.height));
//     box.width = std::max(0, std::min(box.width, shape.width - box.x));
//     box.height = std::max(0, std::min(box.height, shape.height - box.y));
// }

// void clip_boxes(cv::Rect_<float> &box, const cv::Size &shape)
// {
//     box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
//     box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
//     box.width = std::max(0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
//     box.height = std::max(0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
// }

// void clip_boxes(std::vector<cv::Rect> &boxes, const cv::Size &shape)
// {
//     for (cv::Rect &box : boxes)
//     {
//         clip_boxes(box, shape);
//     }
// }

// void clip_boxes(std::vector<cv::Rect_<float>> &boxes, const cv::Size &shape)
// {
//     for (cv::Rect_<float> &box : boxes)
//     {
//         clip_boxes(box, shape);
//     }
// }

// // source: ultralytics/utils/ops.py scale_boxes lines 99+ (ultralytics==8.0.160)
// cv::Rect_<float> scale_boxes(const cv::Size &img1_shape, cv::Rect_<float> &box, const cv::Size &img0_shape, std::pair<float, cv::Point2f> ratio_pad, bool padding)
// {
//     float gain, pad_x, pad_y;

//     if (ratio_pad.first < 0.0f)
//     {
//         gain = std::min(static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
//                         static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width));
//         pad_x = roundf((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
//         pad_y = roundf((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
//     }
//     else
//     {
//         gain = ratio_pad.first;
//         pad_x = ratio_pad.second.x;
//         pad_y = ratio_pad.second.y;
//     }

//     cv::Rect_<float> scaledCoords(box);

//     if (padding)
//     {
//         scaledCoords.x -= pad_x;
//         scaledCoords.y -= pad_y;
//     }

//     scaledCoords.x /= gain;
//     scaledCoords.y /= gain;
//     scaledCoords.width /= gain;
//     scaledCoords.height /= gain;

//     // Clip the box to the bounds of the image
//     clip_boxes(scaledCoords, img0_shape);

//     return scaledCoords;
// }