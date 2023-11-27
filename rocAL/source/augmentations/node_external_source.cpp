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
#include "node_external_source.h"
#include "exception.h"

ExternalSourceNode::ExternalSourceNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs),
                                                                                                                    _file_path(),
                                                                                                                    _dtype(){}

void ExternalSourceNode::create_node() {
    if (_node)
        return;
    vx_array filePathArray = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_CHAR, strlen(_file_path));
    vxAddArrayItems(filePathArray, strlen(_file_path), _file_path, sizeof(char));

    _node = vxExtExternalSource(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), filePathArray, _dtype);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the copy (vxCopyNode) node failed: " + TOSTR(status))
}

void ExternalSourceNode::init(const char* file_path, int dtype) {
    _file_path = new char[strlen(file_path)+1];
    strcpy(_file_path, file_path);
    _file_path[strlen(file_path)] = '\0';
    _dtype = dtype;
}

void ExternalSourceNode::update_node() {
}
