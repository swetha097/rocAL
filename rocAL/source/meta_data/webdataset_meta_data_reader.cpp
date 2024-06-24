/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "meta_data/webdataset_meta_data_reader.h"

#include "pipeline/commons.h"
#include "pipeline/exception.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <libtar.h>
#include <sstream>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

constexpr int create_version_number(int major, int minor, int patch = 0) {
    if (major < 0 || minor < 0 || patch < 0) {
        return -1;
    }
    return major * 1000 + minor * 10 + patch;
}

WebDataSetMetaDataReader::WebDataSetMetaDataReader() {
    _src_dir = nullptr;
    _entity = nullptr;
    _sub_dir = nullptr;
}

void WebDataSetMetaDataReader::init(const MetaDataConfig &cfg,
                                    pMetaDataBatch meta_data_batch) {
    // _path = cfg.path();
    _paths = cfg.path();
    _wds_shards.reserve(_paths.size());
    _index_paths = cfg.index_path();
    _exts = cfg.exts();
    _missing_component_behaviour = cfg.get_missing_component_behaviour();
    _output = meta_data_batch;
}

bool WebDataSetMetaDataReader::exists(const std::string &image_name) {
    return _map_content.find(image_name) != _map_content.end();
}

inline std::tuple<std::string, std::string>
split_name(const std::string &file_path) {
    size_t dot_pos = file_path.find('.', file_path.rfind('/') + 1);
    return {file_path.substr(0, dot_pos), file_path.substr(dot_pos + 1)};
}

void WebDataSetMetaDataReader::add(std::string image_name,
                                   AsciiValues ascii_value) {
    pMetaDataAscii info = std::make_shared<AsciiValue>(ascii_value);
    if (exists(image_name)) {
        auto it = _map_content.find(image_name);
        it->second->get_ascii_values().insert(
            it->second->get_ascii_values().end(), ascii_value.begin(),
            ascii_value.end());
        return;
    }
    _map_content.insert(
        pair<std::string, std::shared_ptr<AsciiValue>>(image_name, info));
}

void WebDataSetMetaDataReader::print_map_contents() {
    std::cerr << "\nMap contents: \n";
    AsciiValues samples_ascii;
    AsciiComponent ascii_component;
    for (auto &elem : _map_content) {
        std::cerr << "Name :\t " << elem.first;
        samples_ascii = elem.second->get_ascii_values();
        for (const auto &sample : samples_ascii) {
            std::cerr << "\n Number of Samples:" << sample.size();
            for (const auto &component_ascii : sample) {
                std::cout << "[" << static_cast<int>(component_ascii) << "]";
                std::cout << " ]" << std::endl;
            }
        }
    }
}

void WebDataSetMetaDataReader::release() { _map_content.clear(); }

void WebDataSetMetaDataReader::release(std::string image_name) {
    if (!exists(image_name)) {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void WebDataSetMetaDataReader::lookup(
    const std::vector<std::string> &image_names) {
    if (image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());
    for (unsigned i = 0; i < image_names.size(); i++) {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_ascii_values_batch()[i] = it->second->get_ascii_values();
    }
}

void WebDataSetMetaDataReader::parse_sample_description(
    std::vector<SampleDescription> &samples_container,
    std::vector<ComponentDescription> &components_container,
    std::ifstream &index_file, const std::string &index_path, int64_t line,
    int index_version) {
    samples_container.emplace_back();
    samples_container.back().components = VectorView<ComponentDescription>(
        components_container, components_container.size());
    samples_container.back().line_number = line;

    // Getting the components data
    std::string components_metadata;
    std::getline(index_file, components_metadata);
    std::stringstream components_stream(components_metadata);

    // Reading consecutive components
    ComponentDescription component;
    while (components_stream >> component.ext) {
        std::cerr << "\n index_version" << index_version;

        if (index_version == create_version_number(1, 2)) {
            if (!(components_stream >> component.offset >> component.size >>
                  component.filename)) {
                THROW("Could not find all necessary component parameters "
                      "(offset, size or filename). Every record in the index "
                      "file should look like: `<ext> <offset> <size> "
                      "<filename>`.");
            }
        } else {
            if (!(components_stream >> component.offset >> component.size))
                THROW("Could not find all necessary component parameters "
                      "(offset or size). Every record in the index file should "
                      "look like: `<ext> <offset> <size>`");
        }

        if (component.filename.empty()) // Use line number as file number
            component.filename = std::to_string(line);

        if (!(component.offset % kBlockSize == 0))
            THROW("tar offset is not a multiple of tar block size kBlockSize, "
                  "perhaps the size value is exported before offset?");

        components_container.emplace_back(std::move(component));
        samples_container.back().components.num++;
    }

    if ((!samples_container.back().components.num))
        THROW("No extensions provided for the sample");
}

inline int parse_index_version(const string &version_str) {
    const char *s = version_str.c_str();
    assert(*s == 'v');
    s++;
    int major = atoi(s);
    s = strchr(s, '.');
    assert(s);
    s++;
    int minor = atoi(s);
    return create_version_number(major, minor);
}

void WebDataSetMetaDataReader::parse_index_files(
    std::vector<SampleDescription> &samples_container,
    std::vector<ComponentDescription> &components_container,
    const std::string &index_path) {
    std::ifstream index_file(index_path);
    std::string global_meta;
    getline(index_file, global_meta);
    std::stringstream global_meta_stream(global_meta);
    std::string index_version_str;
    if (!(global_meta_stream >> index_version_str))
        THROW("Unsupported version of the index file")

    int index_version = parse_index_version(index_version_str);

    int64_t sample_desc_num_signed;
    if (!(global_meta_stream >> sample_desc_num_signed))
        THROW("no sample count found")
    if (!(sample_desc_num_signed > 0))
        THROW("sample count must be positive")

    const size_t sample_desc_num = sample_desc_num_signed;
    samples_container.reserve(samples_container.size() + sample_desc_num);
    for (size_t sample_index = 0; sample_index < sample_desc_num;
         sample_index++) {
        parse_sample_description(samples_container, components_container, index_file,
                        index_path, sample_index + 1, index_version);
    }
}

void WebDataSetMetaDataReader::parse_tar_files(
    std::vector<SampleDescription> &samples_container,
    std::vector<ComponentDescription> &components_container,
    std::unique_ptr<FileIOStream> &tar_file) {
    int64_t initial_file_pos = tar_file->get_current_read_position();
    TarArchive tar_archive(std::move(tar_file));

    std::string last_filename;
    for (; !tar_archive.at_end_of_archive(); tar_archive.advance_to_next_file_in_tar()) {
        if (tar_archive.get_current_file_type() == TarArchive::ENTRY_FILE) {
            std::tie(last_filename, std::ignore) =
                split_name(tar_archive.get_current_file_name());
            break;
        }
    }
    size_t last_components_size = components_container.size();
    for (; !tar_archive.at_end_of_archive(); tar_archive.advance_to_next_file_in_tar()) {
        if (tar_archive.get_current_file_type() != TarArchive::ENTRY_FILE) {
            continue;
        }

        std::string basename, ext;
        std::tie(basename, ext) = split_name(tar_archive.get_current_file_name());
        if (basename.empty()) {
            continue;
        }

        if (basename != last_filename) {
            samples_container.emplace_back();
            samples_container.back().components =
                VectorView<ComponentDescription>(
                    components_container, last_components_size,
                    components_container.size() - last_components_size);
            last_filename = basename;
            last_components_size = components_container.size();
        }

        components_container.emplace_back();
        components_container.back().size = tar_archive.get_current_file_size();
        components_container.back().offset =
            tar_archive.get_current_archive_offset() + tar_archive.get_current_header_size();
        components_container.back().ext = std::move(ext);
        auto _last_id = basename;
        auto last_slash_idx = _last_id.find_last_of("\\/");
        if (std::string::npos != last_slash_idx) {
            _last_id.erase(0, last_slash_idx + 1);
        }
        components_container.back().filename = _last_id;
    }
    samples_container.emplace_back();
    samples_container.back().components = VectorView<ComponentDescription>(
        components_container, last_components_size,
        components_container.size() - last_components_size);

    tar_file = tar_archive.release_file_stream();
}

void WebDataSetMetaDataReader::read_sample_and_add_to_map(
    ComponentDescription component,
    std::unique_ptr<FileIOStream> &current_tar_file_stream,
    AsciiValues ascii_values) {
    if (component.ext != "jpg") { // Add more components as we encounter
        current_tar_file_stream->set_read_position(component.offset);
        std::vector<uint8_t> cls_data(component.size);
        current_tar_file_stream->read_into_buffer(cls_data.data(), component.size);
        AsciiComponent ascii_component;
        for (size_t i = 0; i < cls_data.size(); ++i)
            ascii_component.push_back(static_cast<uint8_t>(cls_data[i]));
        ascii_values.push_back(ascii_component);
        add(component.filename, ascii_values);
    }
}

void WebDataSetMetaDataReader::read_all(const std::string &_path) {

    std::string _folder_path;
    std::string _full_path;
    std::vector<std::string> entry_name_list;
    if (_index_paths.size() == 0) {
        _folder_path = _paths;
        if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
            THROW("ERROR: Failed opening the directory at " + _folder_path);
        _full_path = _folder_path;
        while ((_entity = readdir(_sub_dir)) != nullptr) {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                continue;
            entry_name_list.push_back(entry_name);
        }
        std::sort(entry_name_list.begin(), entry_name_list.end());
        _wds_shards.reserve(entry_name_list.size());
        // Create n such std-streams for n paths
        for (auto &path : entry_name_list)
            _wds_shards.emplace_back(FileIOStream::open(_path + path));
    } else {
        _folder_path = _index_paths;
        if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
            THROW("WebDatasetSourceReader :: ERROR: Failed opening the "
                  "directory at " +
                  _folder_path);
        _full_path = _folder_path;
        while ((_entity = readdir(_sub_dir)) != nullptr) {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                continue;
            index_name_list.push_back(entry_name);
        }
        std::sort(index_name_list.begin(), index_name_list.end());
        if ((_sub_dir = opendir(_path.c_str())) == nullptr)
            THROW("WebDatasetSourceReader :: ERROR: Failed opening the "
                  "directory at " +
                  _path);
        _full_path = _path;
        while ((_entity = readdir(_sub_dir)) != nullptr) {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                continue;
            entry_name_list.push_back(entry_name);
        }
        std::sort(entry_name_list.begin(), entry_name_list.end());
        _wds_shards.reserve(entry_name_list.size());
        for (auto &path : entry_name_list)
            _wds_shards.emplace_back(FileIOStream::open(_path + path));
    }
    closedir(_sub_dir);

    std::vector<SampleDescription> unfiltered_samples;
    std::vector<ComponentDescription> unfiltered_components;

    for (unsigned wds_shard_index = 0; wds_shard_index < entry_name_list.size();
         ++wds_shard_index) {
        unfiltered_samples.resize(0);
        unfiltered_components.resize(0);
        if (_index_paths.size() == 0)
            parse_tar_files(unfiltered_samples, unfiltered_components,
                            _wds_shards[wds_shard_index]);
        else
            parse_index_files(unfiltered_samples, unfiltered_components,
                              _folder_path + index_name_list[wds_shard_index]);

        // After parsing add the contents to the map
        for (auto &sample : unfiltered_samples) {
            AsciiValues ascii_values;
            for (auto &component : sample.components)
                read_sample_and_add_to_map(
                    component, _wds_shards[wds_shard_index], ascii_values);
            ascii_values.clear();
        }
    }
}

void WebDataSetMetaDataReader::read_files(const std::string &_path) {
    if ((_src_dir = opendir(_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;

        std::string file_path = _path;
        file_path.append("/");
        std::string filename(_entity->d_name);
        auto file_extension_idx = filename.find_last_of(".");
        if (file_extension_idx != std::string::npos) {
            std::string file_extension =
                filename.substr(file_extension_idx + 1);
            std::transform(file_extension.begin(), file_extension.end(),
                           file_extension.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if ((file_extension != "jpg") && (file_extension != "jpeg") &&
                (file_extension != "png") && (file_extension != "ppm") &&
                (file_extension != "bmp") && (file_extension != "pgm") &&
                (file_extension != "tif") && (file_extension != "tiff") &&
                (file_extension != "webp") && (file_extension != "wav"))
                continue;
        }
        file_path.append(_entity->d_name);
        _file_names.push_back(file_path);
        _subfolder_file_names.push_back(_entity->d_name);
    }
    if (_file_names.empty())
        WRN("LabelReader: Could not find any file in " + _path)
    closedir(_src_dir);
}
