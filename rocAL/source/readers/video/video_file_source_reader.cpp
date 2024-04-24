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

#include "readers/video/video_file_source_reader.h"

#ifdef ROCAL_VIDEO
VideoFileSourceReader::VideoFileSourceReader() {
    _curr_sequence_idx = 0;
    _loop = false;
    _sequence_id = 0;
    _shuffle = false;
    _sequence_count_all_shards = 0;
}

unsigned VideoFileSourceReader::count_items() {
    if (_loop)
        return _total_sequences_count;
    int ret = (int)(_total_sequences_count - _read_counter);
    return ((ret <= 0) ? 0 : ret);
}

VideoReader::Status VideoFileSourceReader::initialize(ReaderConfig desc) {
    auto ret = VideoReader::Status::OK;
    _sequence_id = 0;
    _folder_path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _video_prop = desc.get_video_properties();
    _video_count = _video_prop.videos_count;
    _video_file_names.resize(_video_count);
    _start_end_frame.resize(_video_count);
    _video_file_names = _video_prop.video_file_names;
    _sequence_length = desc.get_sequence_length();
    _step = desc.get_frame_step();
    _stride = desc.get_frame_stride();
    _video_frame_count = _video_prop.frames_count;
    _start_end_frame = _video_prop.start_end_frame_num;
    _batch_count = desc.get_batch_size();
    _last_batch_info = desc.get_last_batch_policy();
    _stick_to_shard = desc.get_stick_to_shard();
    _shard_size = desc.get_shard_size();
    _total_sequences_count = 0;
    ret = create_sequence_info();

    // shuffle dataset if set
    if (ret == VideoReader::Status::OK && _shuffle)
        std::random_shuffle(_sequences.begin(), _sequences.end());

    return ret;
}

void VideoFileSourceReader::incremenet_read_ptr() {
    _read_counter++;
    _curr_sequence_idx = (_curr_sequence_idx + 1) % _sequences.size();
}

size_t VideoFileSourceReader::last_batch_padded_size() {
    return _last_batch_padded_size;
}

SequenceInfo VideoFileSourceReader::get_sequence_info() {
    auto current_sequence = _sequences[_curr_sequence_idx];
    auto file_path = current_sequence.video_file_name;
    _last_id = file_path;
    incremenet_read_ptr();
    return current_sequence;
}

VideoFileSourceReader::~VideoFileSourceReader() {
}

void VideoFileSourceReader::reset() {
    if (_shuffle) std::random_shuffle(_file_names.begin(), _file_names.end());
    _read_counter = 0;
    _curr_file_idx = 0;
    if (_stick_to_shard == true) {
        _curr_file_idx = 0;
        if (_shard_size > 0) {
            // Reset the variables
            _last_batch_padded_size = _read_counter = _curr_file_idx = 0;
            int read_start_index = 0;
            if (_shard_size < _batch_count)
                read_start_index = _batch_count;
            else {
                if (_shard_size % _batch_count)
                    read_start_index = _shard_size;
                else
                    read_start_index = _shard_size + (_shard_size % _batch_count);
            }
            // To re-arrange the starting index of file-loading keeping in mind the shard_size and batch size
            std::rotate(_file_names.begin(), _file_names.begin() + read_start_index, _file_names.end());
        } else {
            _curr_file_idx = 0;
            _read_counter = 0;
        }
    } else if (_stick_to_shard == false && _shard_count > 1) {
        // Reset the variables
        _last_batch_padded_size = _in_batch_read_count = _curr_file_idx = _file_id = _read_counter = 0;
        _file_names.clear();
        increment_shard_id();
        generate_file_names();  // generates the data from next shard in round-robin fashion after completion of an epoch
        if (_in_batch_read_count > 0 && _in_batch_read_count < _batch_count) {
            // This is to pad within a batch in a shard. Need to change this according to fill / drop or partial.
            // Adjust last batch only if the last batch padded is true.
            replicate_last_image_to_fill_last_shard();
            LOG("FileReader in reset function - Replicated " + _folder_path + _last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count)) + " times to fill the last batch")
        }
        if (!_file_names.empty())
            LOG("FileReader in reset function - Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
    } else {
        _curr_file_idx = 0;
        _read_counter = 0;
    }
}

void VideoFileSourceReader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}

VideoReader::Status VideoFileSourceReader::create_sequence_info() {
    VideoReader::Status status = VideoReader::Status::OK;
    for (size_t i = 0; i < _video_count; i++) {
        unsigned start = std::get<0>(_start_end_frame[i]);
        size_t max_sequence_frames = (_sequence_length - 1) * _stride;
        for (size_t sequence_start = start; (sequence_start + max_sequence_frames) < (start + _video_frame_count[i]); sequence_start += _step) {
            if (get_sequence_shard_id() != _shard_id) {
                _sequence_count_all_shards++;
                incremenet_sequence_id();
                continue;
            }
            _in_batch_read_count++;
            _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
            _sequences.push_back({sequence_start, _video_file_names[i]});
            _last_sequence = _sequences.back();
            _total_sequences_count++;
            _sequence_count_all_shards++;
            incremenet_sequence_id();
        }
    }
    uint sequences_to_pad_shard = _sequence_count_all_shards - (ceil(_sequence_count_all_shards / _shard_count) * _shard_count);
    if (!sequences_to_pad_shard) {
        for (uint i = 0; i < sequences_to_pad_shard; i++) {
            if (get_sequence_shard_id() != _shard_id) {
                _sequence_count_all_shards++;
                incremenet_sequence_id();
                continue;
            }
            _last_sequence = _sequences.at(i);
            _sequences.push_back(_last_sequence);
            _sequence_count_all_shards++;
            incremenet_sequence_id();
        }
    }

    if (_in_batch_read_count > 0 && _in_batch_read_count < _batch_count) {
        replicate_last_sequence_to_fill_last_shard();
        LOG("VideoFileSourceReader ShardID [" + TOSTR(_shard_id) + "] Replicated the last sequence " + TOSTR((_batch_count - _in_batch_read_count)) + " times to fill the last batch")
    }
    if (_sequences.empty())
        WRN("VideoFileSourceReader ShardID [" + TOSTR(_shard_id) + "] Did not load any sequences from " + _folder_path)
    return status;
}

void VideoFileSourceReader::replicate_last_sequence_to_fill_last_shard() {
    if (_last_batch_info.first == RocalBatchPolicy::FILL || _last_batch_info.first == RocalBatchPolicy::PARTIAL) {
        if (_last_batch_info.second == true) {
            for (size_t i = 0; i < (_batch_count - _in_batch_read_count); i++) {
                _sequences.push_back(_last_sequence);
                _total_sequences_count++;
            }
        } else {
            for (size_t i = 0; i < (_batch_count - _in_batch_read_count); i++) {
                _sequences.push_back(_sequences.at(i));
                _total_sequences_count++;
            }
        }
    }
    if (_last_batch_info.first == RocalBatchPolicy::PARTIAL)
        _last_batch_padded_size = _batch_count - _in_batch_read_count;
}

void VideoFileSourceReader::replicate_last_batch_to_pad_partial_shard() {
    if (_sequences.size() >= _batch_count) {
        for (size_t i = 0; i < _batch_count; i++) {
            _sequences.push_back(_sequences[i - _batch_count]);
            _total_sequences_count++;
        }
    }
}

size_t VideoFileSourceReader::get_sequence_shard_id() {
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    // return (_sequence_id / (_batch_count)) % _shard_count;
    return _sequence_id % _shard_count;
}
#endif