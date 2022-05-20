#!/bin/bash
# Updates auto-generated ODS files for the `torch` dialect.
set -euo pipefail

src_dir="$(realpath $(dirname $0)/..)"
build_dir="$(realpath "${TORCH_MLIR_BUILD_DIR:-$src_dir/build}")"
torch_ir_include_dir="${src_dir}/include/torch-mlir/Dialect/Torch/IR"
python_packages_dir="${build_dir}/tools/torch-mlir/python_packages"

extension_pythonpath="/gnn/scs-gnn/exp/custom-nop"
extension_modules="custom_nop"

#ninja -C "${build_dir}"
PYTHONPATH="${python_packages_dir}/torch_mlir:${extension_pythonpath}" python \
  -m torch_mlir.dialects.torch.importer.jit_ir.build_tools.torch_ods_gen \
  --torch_ir_include_dir="${torch_ir_include_dir}" \
  --pytorch_op_extensions=${extension_modules} \
  --debug_registry_dump="${torch_ir_include_dir}/JITOperatorRegistryDump.txt"
