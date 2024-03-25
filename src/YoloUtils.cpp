#include "YoloUtils.h"
#include <stdexcept>

std::vector<std::string> YoloUtils::parseVectorString(const std::string &input)
{
    /* Main purpose of this function is to parse `imgsz` key value of model metadata
     *  and from [height, width] get height, width values in the vector of strings
     * Args:
     *  input:
     *      expected to be something like [544, 960] or [3,544, 960]
     * output:
     *  iterable of strings, representing integers
     */
    std::regex number_pattern(R"(\d+)");

    std::vector<std::string> result;
    std::sregex_iterator it(input.begin(), input.end(), number_pattern);
    std::sregex_iterator end;

    while (it != end)
    {
        result.push_back(it->str());
        ++it;
    }

    return result;
}

std::vector<int> YoloUtils::convertStringVectorToInts(const std::vector<std::string> &input)
{
    std::vector<int> result;

    for (const std::string &str : input)
    {
        try
        {
            int value = std::stoi(str);
            result.push_back(value);
        }
        catch (const std::invalid_argument &e)
        {
            // raise explicit exception
            throw std::invalid_argument("Bad argument (cannot cast): value=" + str);
        }
        catch (const std::out_of_range &e)
        {
            // check bounds
            throw std::out_of_range("Value out of range: " + str);
        }
    }

    return result;
}

std::unordered_map<int, std::string> YoloUtils::parseNames(const std::string &input)
{
    std::unordered_map<int, std::string> result;

    std::string cleanedInput = input;
    cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '{'), cleanedInput.end());
    cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '}'), cleanedInput.end());

    std::istringstream elementStream(cleanedInput);
    std::string element;
    while (std::getline(elementStream, element, ','))
    {
        std::istringstream keyValueStream(element);
        std::string keyStr, value;
        if (std::getline(keyValueStream, keyStr, ':') && std::getline(keyValueStream, value))
        {
            int key = std::stoi(keyStr);
            result[key] = value;
        }
    }

    return result;
}

int64_t YoloUtils::vector_product(const std::vector<int64_t> &vec)
{
    int64_t result = 1;
    for (int64_t value : vec)
    {
        result *= value;
    }
    return result;
}

/**
 * \brief padding value when letterbox changes image size ratio
 */
const int &DEFAULT_LETTERBOX_PAD_VALUE = 114;

void YoloUtils::letterbox(const cv::Mat &image,
                          cv::Mat &outImage,
                          const cv::Size &newShape,
                          cv::Scalar_<double> color,
                          bool auto_,
                          bool scaleFill,
                          bool scaleUp, int stride)
{
    cv::Size shape = image.size();
    float r = std::min(static_cast<float>(newShape.height) / static_cast<float>(shape.height),
                       static_cast<float>(newShape.width) / static_cast<float>(shape.width));
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{r, r};
    int newUnpad[2]{static_cast<int>(std::round(static_cast<float>(shape.width) * r)),
                    static_cast<int>(std::round(static_cast<float>(shape.height) * r))};

    auto dw = static_cast<float>(newShape.width - newUnpad[0]);
    auto dh = static_cast<float>(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = static_cast<float>((static_cast<int>(dw) % stride));
        dh = static_cast<float>((static_cast<int>(dh) % stride));
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = static_cast<float>(newShape.width) / static_cast<float>(shape.width);
        ratio[1] = static_cast<float>(newShape.height) / static_cast<float>(shape.height);
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] || shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }
    else
    {
        outImage = image.clone();
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    if (color == cv::Scalar())
    {
        color = cv::Scalar(DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE);
    }

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void YoloUtils::scaleImage(cv::Mat &scaled_mask, const cv::Mat &resized_mask, const cv::Size &im0_shape, const std::pair<float, cv::Point2f> &ratio_pad)
{
    cv::Size im1_shape = resized_mask.size();

    // Check if resizing is needed
    if (im1_shape == im0_shape)
    {
        scaled_mask = resized_mask.clone();
        return;
    }

    float gain, pad_x, pad_y;

    if (ratio_pad.first < 0.0f)
    {
        gain = std::min(static_cast<float>(im1_shape.height) / static_cast<float>(im0_shape.height),
                        static_cast<float>(im1_shape.width) / static_cast<float>(im0_shape.width));
        pad_x = (im1_shape.width - im0_shape.width * gain) / 2.0f;
        pad_y = (im1_shape.height - im0_shape.height * gain) / 2.0f;
    }
    else
    {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }

    int top = static_cast<int>(pad_y);
    int left = static_cast<int>(pad_x);
    int bottom = static_cast<int>(im1_shape.height - pad_y);
    int right = static_cast<int>(im1_shape.width - pad_x);

    // Clip and resize the mask
    cv::Rect clipped_rect(left, top, right - left, bottom - top);
    cv::Mat clipped_mask = resized_mask(clipped_rect);
    cv::resize(clipped_mask, scaled_mask, im0_shape);
}