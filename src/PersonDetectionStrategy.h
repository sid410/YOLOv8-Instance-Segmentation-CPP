#ifndef PERSON_DETECTION_STRATEGY_H
#define PERSON_DETECTION_STRATEGY_H

#include "IDetectionStrategy.h"
#include <iostream>

class PersonDetectionStrategy : public IDetectionStrategy
{
public:
    std::vector<YoloResults> filterResults(const std::vector<YoloResults> &results) override;
};

#endif
