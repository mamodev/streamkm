
#include <iostream>
#include <fstream>
#include <format>

#include "streamkm/streamkm.hpp"

// 
// #include "core/all.hpp"
// #include "clusterer/all.hpp"
// #include "coreset_stream/all.hpp"

// #include "coreset_reducer/naive.hpp"
// #include "coreset_reducer/dcache.hpp"
// #include "coreset_reducer/swap.hpp"
// #include "coreset_reducer/swap_arena2.hpp"
// #include "coreset_reducer/swap_arena.hpp"
// #include "coreset_reducer/swap_arena_rdist.hpp"
// #include "coreset_reducer/blas_arena_pickpp_2.hpp"
// #include "coreset_reducer/IndexCoresetReducer.hpp"
// #include "coreset_reducer/IndexCachedCoresetReducer3.hpp"
// #include "coreset_reducer/IndexCachedFenwickCoresetReducer.hpp"


using streamkm::Error, streamkm::EResult;
using Metrics = streamkm::CoresetReducerChronoMetrics;
// using Metrics = streamkm::CoresetReducerNoMetrics;
using RandEng = streamkm::xorshift128plus;


EResult<void> Main(int argc, char** argv) {
    rassert(argc == 2 || argc == 3, "Usage: {} <data-folder> [<coreset-size>]", argv[0]);

    streamkm::GnuPerfManager gnu_perf;
    gnu_perf.pause();

    std::string dataset_folder = argv[1];

    std::string data_file = dataset_folder + "/data.bin";
    std::string label_file = dataset_folder + "/labels.bin";

    int coreset_size = std::pow(2, 16); // Default coreset size = 16384
    if (argc == 3) {
        coreset_size = std::atoi(argv[2]);
        rassert(coreset_size > 16, "Coreset size must be greater than 16");
    }


    auto ds_labels = rpropagate(streamkm::read_all_labels(label_file));
        
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* day = std::localtime(&now_c);
    std::string final_print = "";
    size_t stream_bs = coreset_size * 2;
    
    {
        auto ds = rpropagate(streamkm::read_all_dataset(data_file));
        std::cout << "Samples: " << ds.getSamples() << ", Features: " << ds.getFeatures() << std::endl;
        std::cout << "Using stream batch size: " << stream_bs << std::endl;

        auto stream = rpropagate(streamkm::InMemoryDataStream::from_kmds(ds, stream_bs));
        
        std::vector<float> points;
        std::chrono::milliseconds duration;
        {
            auto reducer = streamkm::SwapFCoresetReducer();

            gnu_perf.resume();
            auto start = std::chrono::high_resolution_clock::now(); 
            auto cres = streamkm::coreset_serial_stream(stream, reducer);
            auto centers = rpropagate(cres);
            points = std::move(centers);

            auto end = std::chrono::high_resolution_clock::now();
            gnu_perf.pause();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "SwapArena (outer) total time: " << duration << " ms" << std::endl;
            reducer.metrics.print_avg();

            auto metrics_filename = std::format(".metrics/{:02}_{:02}_{:02}_{}_metrics.csv", day->tm_hour, day->tm_min, day->tm_sec, "swap_fenwick");
            std::ofstream metrics_file(metrics_filename);
            std::string csv = reducer.metrics.toCsv();
            metrics_file << csv;
            metrics_file.close();
            final_print += std::format("python3 an.py {} & \\\n", metrics_filename);
        }

        streamkm::DatasetSamples coreset_ds = {
            .data = points,
            .shape = {static_cast<uint32_t>(coreset_size), static_cast<uint32_t>(ds.getFeatures())}
        };

        streamkm::DatasetSamples ds_dump = {
            .data = ds.data,
            .shape = ds.shape
        };

        rpropagate(streamkm::write_all_dataset(dataset_folder  + "/dataset_dump.bin", ds_dump));
        std::cout << "Dataset dump written to " << dataset_folder   + "/dataset_dump.bin" << std::endl;


        rpropagate(streamkm::write_all_dataset(dataset_folder  + "/coreset_swap_fenwick.bin", coreset_ds));
        std::cout << "Coreset written to " << dataset_folder   + "/coreset_swap_fenwick.bin" << std::endl;

        std::cout << std::format("Kmeans on {} points", points.size() / ds.getFeatures()) << std::endl;
        std::vector<float> centers = kmeans(points.data(), points.size() / ds.getFeatures(), ds.getFeatures(), 3);

        for (std::size_t c = 0; c < 3; ++c) {
            std::cout << "Center " << c << ": ";
            for (std::size_t d = 0; d < ds.getFeatures(); ++d)
                std::cout << centers[c*ds.getFeatures() + d] << " ";
            std::cout << "\n";
        }

        auto fresh_ds = rpropagate(streamkm::read_all_dataset(data_file));

        std::vector<std::size_t> labels = assign_labels(ds.getFeatures(), centers, fresh_ds.data.data(), fresh_ds.getSamples() * fresh_ds.getFeatures());

        rassert(labels.size() == fresh_ds.getSamples(), "Labels size mismatch: expected {}, got {}", fresh_ds.getSamples(), labels.size());
        rassert(ds_labels.size() == fresh_ds.getSamples(), "True labels size mismatch: expected {}, got {}", fresh_ds.getSamples(), ds_labels.size());

        std::size_t min_ds_label = *std::min_element(ds_labels.begin(), ds_labels.end());
        std::size_t max_ds_label = *std::max_element(ds_labels.begin(), ds_labels.end());
        std::size_t min_label = *std::min_element(labels.begin(), labels.end());
        std::size_t max_label = *std::max_element(labels.begin(), labels.end());

        std::cout << "True labels range: [" << min_ds_label << ", " << max_ds_label << "]" << std::endl;
        std::cout << "Pred labels range: [" << min_label << ", " << max_label << "]" << std::endl;
        
        ContingencyMatrix cm(3, 3, ds_labels, labels);
        
        std::cout << "Purity:   " << cm.purity() << std::endl;
        std::cout << "ARI:      " << cm.adjusted_rand_index() << std::endl;
        std::cout << "NMI:      " << cm.normalized_mutual_info() << std::endl;
        std::cout << "FMI:      " << cm.fowlkes_mallows() << std::endl;
    }

    std::cout << "Run the following to plot the results:\n" << final_print << std::endl;

    return {};
}

#include <random>
#include <cblas.h>

int main(int argc, char** argv) {

    std::mt19937 rng;
    std::uniform_int_distribution<int> dist(0, 1000000);
    dist(rng); // Warm up the RNG

    auto res = Main(argc, argv);
    if (res.is_err()) {
        std::cerr << "Error: " << res.error().desc << std::endl;
        return res.error().code != 0 ? res.error().code : 1;
    }


    return 0;
}