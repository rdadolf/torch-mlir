//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertCustomOp
  : public OpConversionPattern<CustomNopOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(CustomNopOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter
                                ) const override {
    // Type checks.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    // Since the Nop does nothing, we simply replace the uses of the nop's
    // return value with its argument, then remove the op.
    op->replaceAllUsesWith(op->getOperands());
    op->erase();

    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateCustomPatternsAndLegality(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<CustomNopOp>();
  patterns.add<ConvertCustomOp>(typeConverter, context);
}
