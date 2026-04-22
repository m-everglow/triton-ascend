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

#ifndef TRITON_SYNC_SCOPE_FLAG_ID_MANAGER_H
#define TRITON_SYNC_SCOPE_FLAG_ID_MANAGER_H

#include "mlir/IR/BuiltinOps.h"
#include <optional>

namespace mlir {
namespace triton {

// --- 公共接口：Flag ID 管理器 ---
// 作用：全局扫描、分配唯一的 flag_id
class FlagIdManager {
public:
  // 构造函数：传入 Module 进行初始化扫描
  FlagIdManager(ModuleOp module);

  // 获取可用的ID，确保不与现有 ID 冲突
  int acquireId();

private:
  // --- 6.2.3.1 扫描现有 Flag ID ---
  // 遍历 Module，找出所有已存在的 flag_id，防止重复分配
  void scanExistingFlags(ModuleOp module);

  // 当前分配的最大 ID
  int64_t currentMaxId = 0;
};

} // namespace triton
} // namespace mlir

#endif //TRITON_SYNC_SCOPE_FLAG_ID_MANAGER_H