#ifndef VEHICLE_DETECTION_STRATEGY_H
#define VEHICLE_DETECTION_STRATEGY_H

#include "IDetectionStrategy.h"
#include <unordered_set>

class VehicleDetectionStrategy : public IDetectionStrategy
{
public:
    std::vector<YoloResults> filterResults(const std::vector<YoloResults> &results) override;
};

#endif
