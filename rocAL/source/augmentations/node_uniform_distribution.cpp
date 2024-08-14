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

#include "augmentations/node_uniform_distribution.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

UniformDistributionNode::UniformDistributionNode(
    const std::vector<Tensor *> &inputs,
    const std::vector<Tensor *> &outputs)
    : Node(inputs, outputs) {}

void UniformDistributionNode::create_node() {
    if (_node) return;
    create_dist_tensor();
    for (uint i = 0; i < _batch_size; i++) {
        update_param();
        float* _float_uniform_distribution_array = static_cast<float*>(_uniform_distribution_array);
        _float_uniform_distribution_array[i] = _dist_uniform(_rngs[i]);
    }
    _outputs[0]->swap_handle(static_cast<void *>(_uniform_distribution_array));
}

void UniformDistributionNode::update_node() {
    for (uint i = 0; i < _batch_size; i++) {
        update_param();
        // Cast void* to float*
        float* _float_uniform_distribution_array = static_cast<float*>(_uniform_distribution_array);
        _float_uniform_distribution_array[i] = _dist_uniform(_rngs[i]);
    }
}

void UniformDistributionNode::update_param() {
    std::uniform_real_distribution<float> dist_uniform(_min, _max);
    _dist_uniform = dist_uniform;
}

// allocate memory for uniform distribution tensor
void UniformDistributionNode::create_dist_tensor() {
    vx_size num_of_dims = 2;
    vx_size stride[num_of_dims];
    std::vector<size_t> _uniform_distribution_array_dims = {_batch_size, 1};
    stride[0] = sizeof(float);
    stride[1] = stride[0] * _uniform_distribution_array_dims[0];
    // vx_enum mem_type = VX_MEMORY_TYPE_HOST;
    // if (_inputs[0]->info().mem_type() == RocalMemType::HIP)
    //     mem_type = VX_MEMORY_TYPE_HIP;
    allocate_host_or_pinned_mem(&_uniform_distribution_array, stride[1] * 4, _inputs[0]->info().mem_type());
}

void UniformDistributionNode::init(std::vector<float> &range) {
    _min = range[0];
    _max = range[1];
    // _uniform_distribution_array.resize(_batch_size);
    BatchRNG<std::mt19937> rng = {ParameterFactory::instance()->get_seed_from_seedsequence(), static_cast<int>(_batch_size)};
    _rngs = rng;
    update_param();
}

UniformDistributionNode::~UniformDistributionNode() {
    if (_inputs[0]->info().mem_type() == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipHostFree(_uniform_distribution_array);
        if (err != hipSuccess)
            std::cerr << "\n[ERR] hipFree failed  " << std::to_string(err) << "\n";
#endif
    } else {
        if (_uniform_distribution_array) free(_uniform_distribution_array);
    }
}

