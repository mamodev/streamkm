#pragma once

#include "errors.hpp"

#include <string>
#include <fstream>
#include <filesystem>
#include <optional>
#include <iostream>
#include <vector>

namespace streamkm
{
    // 5 size because of null terminator
    constexpr uint32_t __str_to_u32_le(const char (&s)[5])
    {
        return (static_cast<uint32_t>(s[3]) << 24) | // Most significant byte
               (static_cast<uint32_t>(s[2]) << 16) |
               (static_cast<uint32_t>(s[1]) << 8) |
               (static_cast<uint32_t>(s[0])); // Least significant byte
    }
    constexpr uint32_t DATASET_FILE_MAGIC = __str_to_u32_le("SKDS");

    struct DatasetFilePrefix
    {
        uint32_t magic;
        uint32_t version;

        static constexpr size_t SIZE = sizeof(magic) + sizeof(version);

        static Result<DatasetFilePrefix, Error> from(std::ifstream &stream)
        {
            DatasetFilePrefix prefix;
            stream.read(reinterpret_cast<char *>(&prefix.magic), sizeof(prefix.magic));
            rassert(stream.gcount() == sizeof(prefix.magic), "Failed to read dataset file magic");
            stream.read(reinterpret_cast<char *>(&prefix.version), sizeof(prefix.version));
            rassert(stream.gcount() == sizeof(prefix.version), "Failed to read dataset file version");
            return prefix;
        }
    };

    struct DatasetFileHeaderV1
    {
        uint64_t file_size;
        uint64_t num_elements;
        uint8_t numeric_format;
        uint8_t flags;
        uint32_t num_dims;

        static constexpr size_t SIZE = sizeof(file_size) + sizeof(num_elements) + sizeof(numeric_format) + sizeof(flags) + sizeof(num_dims);
        static Result<DatasetFileHeaderV1, Error> from(std::ifstream &stream)
        {
            DatasetFileHeaderV1 header;
            stream.read(reinterpret_cast<char *>(&header.file_size), sizeof(header.file_size));
            rassert(stream.gcount() == sizeof(header.file_size), "Failed to read dataset file size");
            stream.read(reinterpret_cast<char *>(&header.num_elements), sizeof(header.num_elements));
            rassert(stream.gcount() == sizeof(header.num_elements), "Failed to read dataset num records");
            stream.read(reinterpret_cast<char *>(&header.numeric_format), sizeof(header.numeric_format));
            rassert(stream.gcount() == sizeof(header.numeric_format), "Failed to read dataset numeric format");
            stream.read(reinterpret_cast<char *>(&header.flags), sizeof(header.flags));
            rassert(stream.gcount() == sizeof(header.flags), "Failed to read dataset flags");
            stream.read(reinterpret_cast<char *>(&header.num_dims), sizeof(header.num_dims));
            rassert(stream.gcount() == sizeof(header.num_dims), "Failed to read dataset num dims");
            return header;
        }
    };

    struct DatasetSamples
    {
        std::vector<float> data;
        std::vector<uint32_t> shape;
        inline size_t getFeatures() const {
            if (shape.size() < 2) return 0;
            
            size_t features = 1;
            for (size_t i = 1; i < shape.size(); i++) {
                features *= shape[i];
            }

            return features;
        }

        inline size_t getSamples() const {
            if (shape.size() < 1) return 0;
            return shape[0];
        }
    };

    EResult<void> write_all_dataset(const std::string &file_path, const DatasetSamples &samples) {
        std::ofstream file(file_path, std::ios::binary);
        rassert(file.is_open(), "Failed to open file for writing: {}", file_path);

        DatasetFilePrefix prefix;
        prefix.magic = DATASET_FILE_MAGIC;
        prefix.version = 1;

        file.write(reinterpret_cast<const char *>(&prefix.magic), sizeof(prefix.magic));
        file.write(reinterpret_cast<const char *>(&prefix.version), sizeof(prefix.version));

        DatasetFileHeaderV1 header;
        header.file_size = DatasetFilePrefix::SIZE + DatasetFileHeaderV1::SIZE + sizeof(uint32_t) * samples.shape.size() + sizeof(float) * samples.data.size();
        header.num_elements = samples.data.size();
        header.numeric_format = 2; // float32
        header.flags = 0;
        header.num_dims = samples.shape.size();

        file.write(reinterpret_cast<const char *>(&header.file_size), sizeof(header.file_size));
        file.write(reinterpret_cast<const char *>(&header.num_elements), sizeof(header.num_elements));
        file.write(reinterpret_cast<const char *>(&header.numeric_format), sizeof(header.numeric_format));
        file.write(reinterpret_cast<const char *>(&header.flags), sizeof(header.flags));
        file.write(reinterpret_cast<const char *>(&header.num_dims), sizeof(header.num_dims));

        for (uint32_t dim : samples.shape) {
            file.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        }

        file.write(reinterpret_cast<const char *>(samples.data.data()), sizeof(float) * samples.data.size());

        rassert(file.good(), "Failed to write dataset to file: {}", file_path);

        return {};
    }
    
    EResult<DatasetSamples> read_all_dataset(const std::string &file_path)
    {
        rassert(std::filesystem::is_regular_file(file_path),
                "File does not exist or is not a regular file: {}", file_path);

        std::ifstream file(file_path, std::ios::binary);
        rassert(file.is_open(), "Failed to open file: {}", file_path);

        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        rassert(size >= DatasetFilePrefix::SIZE + DatasetFileHeaderV1::SIZE,
                "File too small to be a valid dataset file: {}", file_path);

        auto prefix = rpropagate(DatasetFilePrefix::from(file));
        rassert(prefix.magic == DATASET_FILE_MAGIC,
                "Invalid dataset file magic: expected {:08x}, got {:08x}",
                DATASET_FILE_MAGIC, prefix.magic);

        rassert(prefix.version == 1,
                "Unsupported dataset file version: {}", prefix.version);

        auto header = rpropagate(DatasetFileHeaderV1::from(file));
        rassert(header.file_size == static_cast<uint64_t>(size),
                "File size mismatch: expected {}, got {}", header.file_size, size);
        
        rassert(header.numeric_format == 2,
                "Unsupported numeric format: {}", header.numeric_format);

        std::vector<uint32_t> shape(header.num_dims);
        for (uint32_t i = 0; i < header.num_dims; i++)
        {
            file.read(reinterpret_cast<char *>(&shape[i]), sizeof(shape[i]));
            rassert(file.gcount() == sizeof(shape[i]),
                    "Failed to read dimension {} of {}", i, header.num_dims);   
        }

        std::vector<float> data(header.num_elements);
        file.read(reinterpret_cast<char *>(data.data()), sizeof(float) * header.num_elements);
        rassert(file.gcount() == sizeof(float) * header.num_elements,
                "Failed to read dataset data");

        DatasetSamples samples{std::move(data), std::move(shape)};
        return samples;
    }

    // with open(os.path.join(args.outdir, 'labels.bin'), 'wb') as f:
    // Y = np.array(y, dtype=np.uint64)
    // f.write(struct.pack('Q', len(Y.shape)))
    // f.write(Y.tobytes())
    EResult<std::vector<std::size_t>> read_all_labels(const std::string &file_path) {
        rassert(std::filesystem::is_regular_file(file_path),
                "File does not exist or is not a regular file: {}", file_path);

        std::ifstream file(file_path);
        rassert(file.is_open(), "Failed to open file: {}", file_path);  

        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        rassert(size >= sizeof(uint64_t),
                "File too small to be a valid labels file: {}", file_path);

        uint64_t num_dims = 0;
        file.read(reinterpret_cast<char *>(&num_dims), sizeof(num_dims));
        rassert(file.gcount() == sizeof(num_dims),
                "Failed to read number of dimensions from labels file: {}", file_path);

        rassert(num_dims == 1,
                "Unsupported number of dimensions in labels file: {}", num_dims);

        size_t num_labels = (size - sizeof(uint64_t)) / sizeof(uint64_t);

        std::vector<std::size_t> labels(num_labels);
        file.read(reinterpret_cast<char *>(labels.data()), sizeof(uint64_t) * num_labels);
        rassert(file.gcount() == sizeof(uint64_t) * num_labels,
                "Failed to read labels data from file: {}", file_path);


        return labels;
    }
    
} // namespace streamk