# Phase 2: 在线闭环（文件系统）

> 状态: ⬜ 待 Phase 1 结果决定是否启动

## 目标

实现基于文件系统的"穷人版"在线系统 — 推理端产生 trace 文件，训练端消费 trace 文件，新权重通过 checkpoint 回传。

## 架构

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  vLLM Serving (推理节点)      │     │  Training Server (训练节点)   │
│                             │     │                             │
│  vllm serve MiniMax-M2.5    │     │  train_streaming.py         │
│  + Eagle3 Draft             │     │  (FSDP, 8 GPU)             │
│                             │     │                             │
│  Trace Collector ───────────┼─rsync┼─→ Streaming DataLoader     │
│  (hidden_states + logits    │     │                             │
│   + accept/reject mask)     │     │  checkpoint/ ──────────────┤
│                             │     │                             │
│  Weight Reloader ←──────────┼─rsync┼── (poll & reload)          │
│  (poll checkpoint dir)      │     │                             │
└─────────────────────────────┘     └─────────────────────────────┘
```

## 关键组件

### 1. Trace 收集

两种方案：

- **方案 A**: 在 vLLM 投机解码内部 hook（高性能但侵入性强）
- **方案 B**: 旁路 trace 生成（独立进程执行 prefill + 模拟 rejection sampling）

Phase 2 推荐方案 B，复用现有 `VllmHiddenStatesGenerator`。

### 2. 权重热加载

Phase 2 用简单的重启方案：定期检查 checkpoint → 重启 vLLM 进程加载新权重。

### 3. 训练端

完全复用现有 `train_streaming.py` + `sync_datagen.sh` + `manifest.py`。
唯一变化：训练完成后 rsync checkpoint 回推理节点。

## 前置条件

- Phase 1 实验证明 Aurora loss 有效（优于标准 KL）
- 两个空闲节点：1 个推理 (TP=4)，1 个训练 (8 GPU FSDP)

## 预计文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `scripts/data_generation_spec_decode.py` | 新建 | 投机解码 trace 收集 |
| `scripts/weight_reloader.py` | 新建 | Checkpoint poll + reload |
| `scripts/aurora_online.py` | 新建 | 端到端编排 |
| `scripts/sync_checkpoint.sh` | 新建 | Checkpoint 回传脚本 |
