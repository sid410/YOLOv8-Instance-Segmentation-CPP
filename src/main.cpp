#include <filesystem>
#include "OrtApiWrapper/AutoBackendOnnx.h"
#include "OrtApiWrapper/OnnxModelBase.h"
#include "OrtApiWrapper/YoloUtils.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace fs = std::filesystem;

void plotResults(cv::Mat img, std::vector<YoloResults> &results,
                 std::unordered_map<int, std::string> &names,
                 const cv::Size &shape)
{
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type()); // Black background

    for (const auto &res : results)
    {
        float left = res.bbox.x;
        float top = res.bbox.y;

        // Try to get the class name corresponding to the given class_idx
        std::string class_name;
        auto it = names.find(res.class_idx);
        if (it != names.end())
        {
            class_name = it->second;
        }
        else
        {
            std::cerr << "Warning: class_idx not found in names for class_idx = " << res.class_idx << std::endl;
            class_name = std::to_string(res.class_idx);
        }

        // Draw mask if available
        if (res.mask.rows && res.mask.cols > 0)
        {
            mask(res.bbox).setTo(cv::Scalar(255, 255, 255), res.mask); // White color for detected objects
        }
    }

    imshow("mask", mask); // Show only the mask
}

int main()
{
    std::string img_path = "./images/cars.jpg";
    const std::string &modelPath = "./checkpoints/yolov8m-seg.onnx";

    fs::path imageFilePath(img_path);
    fs::path newFilePath = imageFilePath.stem();
    newFilePath += "-kpt-cpp";
    newFilePath += imageFilePath.extension();
    assert(newFilePath != imageFilePath);
    std::cout << "newFilePath: " << newFilePath << std::endl;

    const std::string &onnx_provider = OnnxProviders::CPU; // "cpu";
    const std::string &onnx_logid = "yolov8_inference2";
    float mask_threshold = 0.5f; // in python it's 0.5 and you can see that at ultralytics/utils/ops.process_mask line 705 (ultralytics.__version__ == .160)
    float conf_threshold = 0.30f;
    float iou_threshold = 0.45f; //  0.70f;
    int conversion_code = cv::COLOR_BGR2RGB;

    cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    if (img.empty())
    {
        std::cerr << "Error: Unable to load image" << std::endl;
        return 1;
    }

    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str());
    std::vector<YoloResults> objs = model.predict_once(img, conf_threshold, iou_threshold, mask_threshold, conversion_code);
    std::unordered_map<int, std::string> names = model.getNames();

    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::Size show_shape = img.size();
    plotResults(img, objs, names, show_shape);
    cv::imshow("img", img);
    cv::waitKey();
    return -1;
}
