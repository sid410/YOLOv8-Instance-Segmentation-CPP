#include "VehicleSegmentationMask.h"
#include "PersonSegmentationMask.h"
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

    VehicleSegmentationMask vehicleSegmentation(modelPath);
    cv::Mat vehicleMask = vehicleSegmentation.generateMask(img);
    cv::Mat vehicleDebugImg = vehicleSegmentation.processResultsDebug(img, vehicleMask);

    PersonSegmentationMask personSegmentation(modelPath);
    cv::Mat personMask = personSegmentation.generateMask(img);
    cv::Mat personDebugImg = vehicleSegmentation.processResultsDebug(img, personMask);

    cv::imshow("Segmentation Mask", personMask);
    cv::imshow("Debug Image", personDebugImg);

    cv::waitKey(0);
    return 0;
}
