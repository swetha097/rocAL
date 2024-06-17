/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include <dirent.h>

#include <map>

#include "pipeline/commons.h"
#include "meta_data/meta_data.h"
#include "meta_data/meta_data_reader.h"
#include "readers/image/image_reader.h"
#include "pipeline/filesystem.h"
#include <unordered_set>
#include <set>
#include "tar_utils.h"
class WebDataSetMetaDataReader : public MetaDataReader {
   public:
    void init(const MetaDataConfig& cfg, pMetaDataBatch meta_data_batch) override;
    void lookup(const std::vector<std::string>& image_names) override;
    void read_all(const std::string& path) override;
    void release(std::string image_name);
    void release() override;
    void print_map_contents();
    bool set_timestamp_mode() override { return false; }
    const std::map<std::string, std::shared_ptr<MetaData>>& get_map_content() override { return _map_content; }

    WebDataSetMetaDataReader();

   private:
    void read_files(const std::string& _path);
    bool exists(const std::string& image_name) override;
    void add(std::string image_name, int label);
    std::map<std::string, std::shared_ptr<MetaData>> _map_content;
    std::map<std::string, std::shared_ptr<MetaData>>::iterator _itr;
    std::string _path;
    std::string _paths, _index_paths;
    unsigned _missing_component_behaviour;
    pMetaDataBatch _output;
    DIR *_src_dir, *_sub_dir;
    struct dirent* _entity;
    std::vector<std::string> _file_names;
    std::vector<std::set<std::string>> _exts;
    std::vector<std::string> _subfolder_file_names;
    void parse_tar_files(std::vector<SampleDescription>& samples_container,
                                              std::vector<ComponentDescription>& components_container,
                                              std::unique_ptr<StdFileStream>& tar_file);
    void parse_index_files(std::vector<SampleDescription>& samples_container,
                                              std::vector<ComponentDescription>& components_container,
                                              const std::string& paths_to_index_files);
    void read_sample_and_add_to_map(ComponentDescription component, std::unique_ptr<StdFileStream>& current_tar_file_stream);
    const std::string kCurrentIndexVersion = "v1.2";  // NOLINT
    std::vector<std::unique_ptr<StdFileStream>> _wds_shards;
    const std::unordered_set<std::string> kSupportedIndexVersions = {"v1.1", kCurrentIndexVersion};
    std::tuple<std::string, std::string> split_name(const std::string& file_path) ;
};
