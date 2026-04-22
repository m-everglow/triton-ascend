/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonSyncScope/Common/FlagIdManager.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flag-id-manager"

namespace mlir {
namespace triton {

FlagIdManager::FlagIdManager(ModuleOp module) {
  scanExistingFlags(module);
  LLVM_DEBUG(llvm::dbgs() << "FlagIdManager: Initialized with max_id = " << currentMaxId << "\n");
}

void FlagIdManager::scanExistingFlags(ModuleOp module) {
  module.walk([&](Operation *op) {
    // 同步指令有 "flag" 属性，如: flag = 5
    if (auto attr = op->getAttr("flag")) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        currentMaxId = std::max(currentMaxId, (int64_t)intAttr.getInt());
      }
    }
    // 也检查 static_flag_id 属性（用于双缓冲场景）
    if (auto attr = op->getAttr("static_flag_id")) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        currentMaxId = std::max(currentMaxId, (int64_t)intAttr.getInt());
      }
    }
  });
}

int FlagIdManager::acquireId() {
  // 6.2.3.1 & 6.2.3.2 逻辑：
  // 简单策略：直接递增。
  // 如果需要复用逻辑，可以在这里添加基于 CFG 的分析，或者返回一个空闲池中的 ID。
  // 目前为了保证全局唯一且简单，采用递增策略。
  return ++currentMaxId;
}

} // namespace triton
} // namespace mlir