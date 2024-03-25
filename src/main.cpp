#include <random>

#include <filesystem>
#include "nn/onnx_model_base.h"
#include "nn/autobackend.h"
#include "YoloUtils.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace fs = std::filesystem;

cv::Scalar generateRandomColor(int numChannels)
{
    if (numChannels < 1 || numChannels > 3)
    {
        throw std::invalid_argument("Invalid number of channels. Must be between 1 and 3.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

    cv::Scalar color;
    for (int i = 0; i < numChannels; i++)
    {
        color[i] = dis(gen); // for each channel separately generate value
    }

    return color;
}

std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels)
{
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < class_names_num; i++)
    {
        cv::Scalar color = generateRandomColor(numChannels);
        colors.push_back(color);
    }
    return colors;
}

void plot_results(cv::Mat img, std::vector<YoloResults> &results,
                  std::vector<cv::Scalar> color, std::unordered_map<int, std::string> &names,
                  const cv::Size &shape)
{
    cv::Mat mask = img.clone();

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
            // Then convert it to a string anyway
            class_name = std::to_string(res.class_idx);
        }

        // Draw mask if available
        if (res.mask.rows && res.mask.cols > 0)
        {
            mask(res.bbox).setTo(color[res.class_idx], res.mask);
        }

        // Create label
        std::stringstream labelStream;
        labelStream << class_name << " " << std::fixed << std::setprecision(2) << res.conf;
        std::string label = labelStream.str();

        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
        cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
        rectangle(img, rect_to_fill, color[res.class_idx], -1);
        putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
    }

    // Combine the image and mask
    addWeighted(img, 0.6, mask, 0.4, 0, img);
    imshow("mask", mask);
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
    std::vector<cv::Scalar> colors = generateRandomColors(model.getNc(), model.getCh());
    std::unordered_map<int, std::string> names = model.getNames();

    std::vector<std::vector<float>> keypointsVector;
    for (const YoloResults &result : objs)
    {
        keypointsVector.push_back(result.keypoints);
    }

    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::Size show_shape = img.size();
    plot_results(img, objs, colors, names, show_shape);
    cv::imshow("img", img);
    cv::waitKey();
    return -1;
}
