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

#include "loaders/audio/audio_source_evaluator.h"

#include "decoders/audio/audio_decoder_factory.hpp"
#include "readers/image/reader_factory.h"

#ifdef ROCAL_AUDIO

size_t AudioSourceEvaluator::GetMaxSamples() {
    return _samples_max;
}

size_t AudioSourceEvaluator::GetMaxChannels() {
    return _channels_max;
}

AudioSourceEvaluatorStatus
AudioSourceEvaluator::Create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg) {
    AudioSourceEvaluatorStatus status = AudioSourceEvaluatorStatus::OK;
    // Can initialize it to any decoder types if needed
    _decoder = create_audio_decoder(std::move(decoder_cfg));
    _reader = create_reader(std::move(reader_cfg));
    FindMaxDimension();
    return status;
}

void AudioSourceEvaluator::FindMaxDimension() {
    _reader->reset();
    while (_reader->count_items()) {
        size_t fsize = _reader->open();
        if (!fsize) continue;
        auto file_name = _reader->file_path();
        if (file_name.find("train") != std::string::npos) {
            _samples_max = 475760;
            _channels_max = 1;
        } else if (file_name.find("val") != std::string::npos) {
            _samples_max = 522320;
            _channels_max = 1;
        }
        break;
    }
    _reader->reset();
}
#endif
