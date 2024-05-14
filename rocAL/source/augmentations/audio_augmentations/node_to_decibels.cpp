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

#include "augmentations/audio_augmentations/node_to_decibels.h"

#include <vx_ext_rpp.h>

#include "pipeline/exception.h"

ToDecibelsNode::ToDecibelsNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void ToDecibelsNode::create_node() {
    if (_node)
        return;

    vx_status status = VX_SUCCESS;
    RocalAudioAugmentation _augmentation_enum = ROCAL_TO_DECIBELS;
    vx_scalar augmentation_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_augmentation_enum);
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);

    vx_array float_values_vx = vxCreateArray(
        vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, 3);
    status = vxAddArrayItems((vx_array)float_values_vx, 1, &_cutoff_db,
                             sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the to_decibels node: " +
              TOSTR(status))
    status = vxAddArrayItems((vx_array)float_values_vx, 1, &_multiplier,
                             sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the to_decibels node: " +
              TOSTR(status))
    status = vxAddArrayItems((vx_array)float_values_vx, 1, &_reference_magnitude,
                             sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the to_decibels node: " +
              TOSTR(status))
    _node = vxExtRppAudioNodes(
        _graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), nullptr,
        _inputs[0]->get_roi_tensor(), _outputs[0]->get_roi_tensor(),
        nullptr, float_values_vx, input_layout_vx, output_layout_vx,
        nullptr, nullptr, augmentation_type_vx);
    
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the to_decibels (vxRppToDecibels) node failed: " + TOSTR(status))
}

void ToDecibelsNode::update_node() {}

void ToDecibelsNode::init(float cutoff_db, float multiplier, float reference_magnitude) {
    _cutoff_db = cutoff_db;
    _multiplier = multiplier;
    _reference_magnitude = reference_magnitude;
}
