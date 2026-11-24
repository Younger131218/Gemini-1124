# IPCR v2 — 室内分类点云逆向建模（干净模块化主干）

面向“已分类室内 LAS 点云 → 闭合建筑外壳 + 语义模型库家具 + 规则杂物 → 带顶板类色 Z-up / Y-up 配对 GLB”的干净模块化重建主干。

- **唯一底线规范**：[最终底层基线要求 v1.1](docs/最终底层基线要求_v1.1.md)（只读 hash pin 校验，AI 不得篡改）
- **架构说明**：[ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **当前状态**：[STATUS.md](STATUS.md)

## 快速命令

```powershell
# 1. 核验冻结底线 v1.1
.\.venv\Scripts\python.exe -m ipcr.cli verify-baseline

# 2. 运行全部单元测试（不依赖私有数据）
.\.venv\Scripts\python.exe -m pytest

# 3. 审计只读模型库并生成脱敏元数据表
.\.venv\Scripts\python.exe -m ipcr.cli audit-assets --library "<模型库路径>"

# 4. 预检单个 LAS 文件
.\.venv\Scripts\python.exe -m ipcr.cli preflight --las "<LAS路径>"

# 5. 运行端到端重建（阶段 1 建筑外壳）
.\.venv\Scripts\python.exe -m ipcr.cli run --las "<LAS路径>" --stage arch --out "runs/<run_id>"
```

## 仓库边界
本仓库仅保存自有源码、脱敏配置、资产元数据与单元测试；绝不提交原始 LAS、输出 GLB、模型库二进制、点云缓存或现场截图。
