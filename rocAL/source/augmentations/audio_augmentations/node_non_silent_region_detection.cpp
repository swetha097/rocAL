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

#include "augmentations/audio_augmentations/node_non_silent_region_detection.h"
#include <vx_ext_rpp.h>
#include "pipeline/exception.h"

NonSilentRegionDetectionNode::NonSilentRegionDetectionNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void NonSilentRegionDetectionNode::create_node() {
    if (_node)
        return;

    vx_status status;
    RocalAudioAugmentation _augmentation_enum = ROCAL_NON_SILENT_REGION_DETECTION;
    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    vx_scalar augmentation_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_augmentation_enum);
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_array int_values_vx = vxCreateArray(
        vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, 2);
    status = vxAddArrayItems((vx_array)int_values_vx, 1, &_window_length,
                             sizeof(vx_int32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the pre-emphasis filter node: " +
              TOSTR(status))
    status = vxAddArrayItems((vx_array)int_values_vx, 1, &_reset_interval,
                             sizeof(vx_int32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the pre-emphasis filter node: " +
              TOSTR(status))

    vx_array float_values_vx = vxCreateArray(
        vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, 2);
    status = vxAddArrayItems((vx_array)float_values_vx, 1, &_cutoff_db,
                             sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the pre-emphasis filter node: " +
              TOSTR(status))
    status = vxAddArrayItems((vx_array)float_values_vx, 1, &_reference_power,
                             sizeof(vx_float32));
    if (status != 0)
        THROW(" vxAddArrayItems failed in the pre-emphasis filter node: " +
              TOSTR(status))
    
    _node = vxExtRppAudioNodes(
        _graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _outputs[1]->handle(),
        _inputs[0]->get_roi_tensor(), _outputs[0]->get_roi_tensor(),
        int_values_vx, float_values_vx, input_layout_vx, output_layout_vx,
        nullptr, nullptr, augmentation_type_vx);
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW(
            "Adding the vxExtRppAudioNodes - PreemphasisFilter node failed: " +
            TOSTR(status))
}

void NonSilentRegionDetectionNode::update_node() {}

void NonSilentRegionDetectionNode::init(float cutoff_db, float reference_power, int window_length, int reset_interval) {
    _cutoff_db = cutoff_db;
    _reference_power = reference_power;
    _window_length = window_length;
    _reset_interval = reset_interval;
}
