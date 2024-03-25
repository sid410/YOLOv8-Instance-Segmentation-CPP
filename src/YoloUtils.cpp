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
