#include "PersonDetectionStrategy.h"

std::vector<YoloResults> PersonDetectionStrategy::filterResults(const std::vector<YoloResults> &results)
{
    std::vector<YoloResults> filtered;
    std::copy_if(results.begin(), results.end(), std::back_inserter(filtered),
                 [](const YoloResults &result)
                 {
                     return result.class_idx == 0; // Person class
                 });
    std::cout << "Detected People: " << filtered.size() << std::endl;
    return filtered;
}
