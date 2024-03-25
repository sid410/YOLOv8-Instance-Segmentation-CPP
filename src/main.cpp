#include "VehicleSegmentationMask.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    std::string imgPath = "./images/cars.jpg";
    const std::string modelPath = "./checkpoints/yolov8m-seg.onnx";

    cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    if (img.empty())
    {
        std::cerr << "Error: Unable to load image" << std::endl;
        return 1;
    }

    VehicleSegmentationMask segmentation(modelPath);
    cv::Mat mask = segmentation.generateMask(img);
    cv::Mat debugImg = segmentation.processResultsDebug(img, mask);

    cv::imshow("Segmentation Mask", mask);
    cv::imshow("Debug Image", debugImg);

    cv::waitKey(0);
    return 0;
}
