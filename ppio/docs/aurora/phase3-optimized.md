# Phase 3: 优化版在线系统

> 状态: ⬜ 待 Phase 2 验证后启动

## 目标

在 Phase 2 基础上优化性能，接近论文描述的完整系统。

## 优化方向

### 1. vLLM 内部 Trace Hook

替换 Phase 2 旁路方案，直接在投机解码验证步骤中收集 trace。
零额外推理开销（复用已有 verify forward pass）。

### 2. GPU 内存 Buffer

替换文件系统为 GPU 内存 buffer，用 `torch.distributed.rpc` 实现跨节点传输。

### 3. 真正的 Hot-swap

无重启权重更新：直接修改 worker model state_dict。

### 4. Lazy Sync 策略

论文的 lazy sync：每 N 个请求同步一次，减少同步开销。

### 5. Tree Attention

扩展 flex_attention 支持树形分支掩码，在一次前向传播中同时处理 accepted 和 rejected 分支。

## 前置条件

- Phase 2 在线闭环运行稳定
- 性能瓶颈已通过 profiling 明确
