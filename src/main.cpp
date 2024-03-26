#include "SegmentationMask.h"
#include "VehicleDetectionStrategy.h"
#include "PersonDetectionStrategy.h"
#include <opencv2/opencv.hpp>
#include <memory>

int main()
{
    std::string imgPath = "./images/team.jpg";
    const std::string modelPath = "./checkpoints/yolov8n-seg.onnx";

    cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    if (img.empty())
    {
        std::cerr << "Error: Unable to load image" << std::endl;
        return 1;
    }

    // Choose detection strategy based on the need (vehicle or person)
    // std::unique_ptr<IDetectionStrategy> strategy = std::make_unique<VehicleDetectionStrategy>();
    std::unique_ptr<IDetectionStrategy> strategy = std::make_unique<PersonDetectionStrategy>();

    SegmentationMask segmentation(modelPath, std::move(strategy));
    cv::Mat mask = segmentation.generateMask(img);
    cv::Mat highlightedImg = segmentation.processResultsDebug(img, mask);

    // Display results
    cv::imshow("Mask", mask);
    cv::imshow("Highlighted Image", highlightedImg);
    cv::waitKey(0);

    return 0;
}
