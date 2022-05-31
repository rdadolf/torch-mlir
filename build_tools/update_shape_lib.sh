#!/bin/bash
# Updates auto-generated shape library files for the `torch` dialect.
set -e

src_dir="$(realpath $(dirname $0)/..)"
build_dir="$(realpath "${TORCH_MLIR_BUILD_DIR:-$src_dir/build}")"
torch_transforms_cpp_dir="${src_dir}/lib/Dialect/Torch/Transforms"
python_packages_dir="${build_dir}/tools/torch-mlir/python_packages"

extension_pythonpath="/gnn/scs-gnn/build/exp"
extension_modules="custom_nop"

#ninja -C "${build_dir}"
PYTHONPATH="${python_packages_dir}/torch_mlir:${extension_pythonpath}" python \
  -m torch_mlir.dialects.torch.importer.jit_ir.build_tools.shape_lib_gen \
  --pytorch_op_extensions=${extension_modules} \
  --torch_transforms_cpp_dir="${torch_transforms_cpp_dir}"
