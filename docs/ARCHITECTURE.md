# IPCR v2 模块架构与接口契约

本文定义各模块的数据边界，确保 Python 原型可清晰、逐模块平滑移植至 C++17 动态库（C ABI 1.0）。

## 分层流水线

1. **输入层 (`io_las.py`)**：只读加载 LAS 1.2–1.4，计算 SHA-256，使用 `float64` 扣除局部原点（`origin_xyz`），避免 `float32` 大坐标精度损失。
2. **工作标签层 (`working_labels.py`)**：原标签 `raw_labels` 只读；输出双轨 `WorkingLabelState(raw_labels, work_labels, rule_ids, audit_records)`。规则包括 WL-A（高位平顶 `0→11`）、WL-B（物理不可能微噪点隔离）、WL-C1（墙柱贴平混分）、WL-C2（家具杂物混分，阶段 2/3），模糊簇保守保留。
3. **建筑闭合外壳层 (`architecture/`)**：
   - `frame_floors.py`：场景主轴估计、多楼层地面/顶板高度层、无内部贯穿孔的楼板/顶板域；
   - `walls_shell.py`：主墙面与隔墙提取、L/T/X 墙角拓扑接头、B13a 最外围连续闭合外壳（外墙脚下延接地板 + 地板外延接墙脚）；
   - `openings.py`：B14 门/窗/无门通道身份裁决与物理宿主家族统一精确扣洞；
   - `beams_columns.py` & `stairs_railings.py`：梁、真柱实体与楼梯/平台/扶手证据触发框架。
4. **家具与杂物层（阶段 2/3 预留）**：以 `ArchShell` 作为硬边界约束输入，执行类内实例拆分、模型库语义部件配准与联合防穿模。
5. **导出层 (`export_glb.py`)**：严格遵循 R23 契约（B19–B21）——带顶板类色 `Z-up` 与 `Y-up` 配对，1 Scene / 2 Nodes / 1 Mesh / 每输出类 1 Primitive，父根 Z-up 单位阵、Y-up 标准右手 `Rx(-90°)`，子节点携带 `origin_xyz` 补偿平移。
6. **独立验证层 (`validator/`)**：从磁盘独立回读最终 Z/Y GLB，完成格式契约、右手性、闭合正体积、零面积/非流形、真实三角网格相交及双向点云覆盖率核验。

## 配置原则
- 全程通过强类型 `PipelineConfig` 显式传参，禁止环境变量隐式缓存或全局可变单例。
- 禁止硬编码任何场景名称、坐标、实例编号或固定对象数量。
