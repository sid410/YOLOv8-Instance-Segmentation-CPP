#ifndef I_DETECTION_STRATEGY_H
#define I_DETECTION_STRATEGY_H

#include "OrtApiWrapper/YoloUtils.h"
#include "OrtApiWrapper/AutoBackendOnnx.h"
#include <vector>

class IDetectionStrategy
{
public:
    virtual std::vector<YoloResults> filterResults(const std::vector<YoloResults> &results) = 0;
    virtual ~IDetectionStrategy() = default;
};

#endif
