#pragma once

#include "streamkm/core/errors.hpp"
#include "streamkm/core/parser.hpp"

#include <concepts> 
#include <span>

namespace streamkm {

template<typename T>
concept DataStream = requires(T t) {
    { t.next() } -> std::same_as<std::span<float>>;
    { t.has_next() } -> std::same_as<bool>;
    { t.getFeatures() } -> std::same_as<size_t>;
    { T::thread_safe } -> std::convertible_to<bool>;
};

class InMemoryDataStream
{
private:
    std::span<float> data;
    size_t features;
    size_t position;
    size_t total_samples;
    size_t batch_size;

    InMemoryDataStream(std::span<float> data, size_t features, size_t batch_size)
        : data(data), features(features), position(0), total_samples(data.size() / features), batch_size(batch_size)
    {}


public:
    static constexpr bool thread_safe = false;

    static EResult<InMemoryDataStream> from_kmds(DatasetSamples& ds, size_t batch_size = 1024) {
        
        rassert(ds.getFeatures() > 0,   "Dataset has zero features");
        rassert(ds.getSamples() > 0,    "Dataset has zero samples");
        rassert(batch_size > 2,         "Batch size must be greater than two");
        rassert(batch_size % 2 == 0,    "Batch size must be even");

        return InMemoryDataStream(
            std::span<float>(ds.data.data(), ds.data.size()),
            ds.getFeatures(),
            batch_size
        );
    }

    std::span<float> next() {
        if(!has_next()) return {};
        size_t batch_size = std::min(this->batch_size, total_samples - position);
        auto res = data.subspan(position * features, batch_size * features);
        position += batch_size;
        return res;
    }

    bool has_next() const {
        return position < total_samples;
    }

    size_t getFeatures() const {
        return features;
    }
};
static_assert(DataStream<InMemoryDataStream>);

}