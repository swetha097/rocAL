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

#include <vx_ext_rpp.h>
#include "node_brightness.h"
#include "exception.h"

BrightnessNode::BrightnessNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs),
                                                                                                            _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1]),
                                                                                                            _beta(BETA_RANGE[0], BETA_RANGE[1]) {}

void BrightnessNode::create_node() {
    if (_node)
        return;

    if(_tensor_alpha && _tensor_alpha->info().is_external_source()) {
        _alpha.set_tensor(_tensor_alpha->handle());
    } else {
        _alpha.create_tensor(_graph, VX_TYPE_FLOAT32, _batch_size);
    }

    if(_tensor_beta && _tensor_beta->info().is_external_source()) {
        _beta.set_tensor(_tensor_beta->handle());
    } else {
        _beta.create_tensor(_graph, VX_TYPE_FLOAT32, _batch_size);
    }

    int input_layout = static_cast<int>(_inputs[0]->info().layout());
    int output_layout = static_cast<int>(_outputs[0]->info().layout());
    int roi_type = static_cast<int>(_inputs[0]->info().roi_type());
    vx_scalar input_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar output_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);

    _node = vxExtRppBrightness(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _alpha.default_tensor(), _beta.default_tensor(), input_layout_vx, output_layout_vx,roi_type_vx);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the brightness (vxExtRppBrightness) node failed: " + TOSTR(status))
}

void BrightnessNode::init(float alpha, float beta) {
    _alpha.set_param(alpha);
    _beta.set_param(beta);
}

void BrightnessNode::init(Tensor *alpha, Tensor *beta) {
    _tensor_alpha = alpha;
    _tensor_beta = beta;
    if (_tensor_alpha && _tensor_alpha->info().is_external_source() == false) {
        _alpha.set_param(core(std::get<FloatParam*>(_tensor_alpha->get_param())));
    }
    if (_tensor_beta && _tensor_beta->info().is_external_source() == false) {
        _beta.set_param(core(std::get<FloatParam*>(_tensor_beta->get_param())));
    }
}

void BrightnessNode::update_node() {
    if (_tensor_alpha && _tensor_alpha->info().is_external_source() == false)
        _alpha.update_tensor();
    if (_tensor_beta && _tensor_beta->info().is_external_source() == false)
        _beta.update_tensor();
}
