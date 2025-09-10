#pragma once

#include "core/all.hpp"
#include "ds_stream.hpp"

#include "coreset_reducer/coreset_reducer.hpp"

#include <concepts>
#include <span>
#include <vector>
#include <cstddef> // for size_t

#include <iostream>

namespace streamkm {

template<CoresetReducer Reducer>
EResult<h_rr_t<Reducer>> coreset_serial_stream(DataStream auto& stream) {

    size_t d = stream.getFeatures();
    rassert(d > 0, "Data stream returned zero features");

    std::vector<std::optional<h_rr_t<Reducer>>> buckets;

    size_t batch_id = 0;
    while (stream.has_next()) {
        auto data = stream.next();
        rassert(data.size() % d == 0, "Data stream returned data size not divisible by features");
        rassert(data.size() / d > 0, "Batch of data stream should contain at least two samples");

        batch_id++;

        size_t n = data.size() / d;
        auto coreset = Reducer::reduce(data.data(), n, d, n / 2);
        // std::cout << "Processed batch " << batch_id << std::endl;

        size_t next = 0;

        while(buckets.size() > next && buckets[next].has_value()) {
            coreset = std::move(Reducer::reduce(coreset, buckets[next].value()));
            buckets[next].reset();
            next++;
        }

        if(next >= buckets.size()) {
            buckets.push_back(std::move(coreset));
        } else {
            buckets[next] = std::move(coreset);
        }
    }

    // Final merge
    size_t pos = 0;
    while(pos + 1 < buckets.size()) {
        if (buckets[pos].has_value())
            break;

        pos++;
        continue;
    }

    rassert(pos < buckets.size(), "No data in stream");

    auto final_coreset = std::move(buckets[pos].value());
    if (pos + 1 == buckets.size()) {
        return std::move(final_coreset);
    }

    pos++;
    while(pos < buckets.size()) {
        if (!buckets[pos].has_value()) {
            pos++;
            continue;
        }

        final_coreset = std::move(
            Reducer::reduce(final_coreset, buckets[pos].value())
        );

        buckets[pos].reset();
        pos++;
    }

    return std::move(final_coreset);
}

}

