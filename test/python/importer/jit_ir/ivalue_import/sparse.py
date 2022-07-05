# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # (0,0):1; (0,1):2; (2,2):3
        values = torch.tensor([1,2,3])
        self.coo = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 2],
                          [0, 1, 2]]),
            values,
            dtype=torch.float32)

        #row_idx = torch.tensor([0,2,3])
        #col_idx = torch.tensor([0,1,2])
        #self.sp = torch.sparse_csr_tensor(
        #  torch.tensor([0, 2, 3]),
        #  torch.tensor([0, 1, 2]),
        #  values,
        #  dtype=torch.float32)

# CHECK: %[[COO:.*]] = torch.tensor.literal(sparse<{{\[\[}}0, 0, 2], [0, 1, 2]], [1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3x3xf32>) : !torch.tensor<[3,3],f32>
# FIXME # CHECK: %[[CSR:.*]] = torch.tensor.literal(sparse<{{\[\[}}0, 0, 2], [0, 1, 2]], [1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3x3xf32>) : !torch.tensor<[3,3],f32>
# CHECK: %[[ROOT:.*]] = torch.nn_module  {
# CHECK:   torch.slot "coo", %[[COO]] : !torch.tensor<[3,3],f32>
# FIXME # CHECK:   torch.slot "csr", %[[CSR]] : !torch.tensor<[3,3],f32>
# CHECK: }


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
