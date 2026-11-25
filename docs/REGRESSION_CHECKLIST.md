# IPCR v2 永久防复发门禁与防上下文丢失自审清单（G1–G12）

> **目的**：本文件常驻仓库 `docs/REGRESSION_CHECKLIST.md`，配合 `run_report.json` 中的 `invariants_audit`、自动化单元测试 `tests/` 与四场景门禁脚本 `scripts/audit_scenes.py`，确保即使在长对话上下文压缩或换人接手后，**任何历史已解决缺陷都无法静默复发**。

## 强制交付前自审协议（每次修改算法后必须执行）

1. **先读本清单**：修改任何模块前，先核对受影响的 `G1–G12` 门禁条款。
2. **全量单测**：运行 `.venv\Scripts\python.exe -X utf8 -m pytest -v`，必须 **100% PASS**。
3. **四场景全量回归**：对 `TJHerlou.las`、`办公室.las`、`DCJSsysj2.las`、`20261010一楼大厅.las` 跑 `stage="full"`（使用全新 `run_id`，严禁覆盖历史目录）。
4. **机器门禁审校**：运行 `.venv\Scripts\python.exe -X utf8 scripts\audit_scenes.py <4 个最新 run 目录>`，必须输出 **`ALL 4 SCENES INVARIANTS PASS`**（退出码 0）。
5. **视觉叠加图复查**：逐一人工查看全部 4 张 `review_overlay.png` 与重点截面指标，确认无新增视觉回归后才向用户交付。

---

## 全量防复发门禁表（G1–G12）

| 门禁 | 历史缺陷来源 | 永久代码约束与位置 | 自动检验方式 |
|---|---|---|---|
| **G1** | 门/窗与墙体穿模、敞开门扇缺失 | B13a/B14 墙洞剖分与敞开门闭合归位（`walls_shell.py`, `openings.py`） | `relation_audit == PASS` (`pen == 0`) + `test_open_swing_door_and_wall_void_door_rules` |
| **G2** | `DCJS` 方桌 PCA 倾斜 | 2D 凸包旋转卡壳最小面积 OBB（`furniture/tables_cabinets_sofas.py`） | `test_stage2_furniture.py` |
| **G3** | `metal_filing_cabinet` 模型库顶/背空洞 | 全图元网格修补 + 6 面实心门禁（`furniture/library.py`） | `test_stage2_furniture.py` + `glb_validation == PASS` |
| **G4** | `办公室` 东排椅 13–21 cm 偏置与幽灵第 5 把椅 | 4 cm 体素均匀化中点 + 1D 靠背谷底分椅（`furniture/chairs.py`） | `test_stage2_furniture.py` |
| **G5** | `TJHerlou` 单椅点云重叠种两把椅 | 2D 上身非重叠 + `≥ 0.48 m` 中心距 + `0.45 m` 碎块去重（`furniture/chairs.py`） | `test_stage2_furniture.py` |
| **G6** | 桌面边缘噪点假 L 缺口与异形桌 | 桌面边缘滤波 + 凸包/异形桌支持（`furniture/tables_cabinets_sofas.py`） | `test_stage2_furniture.py` |
| **G7** | `DCJS` 半空悬空黄锥与游离噪点杂物 | 4 类真实物理支撑门禁 + 锥体 `≥ 220°` 环绕 + 植物/行人排除（`clutter/regular_clutter.py`） | `test_stage3_clutter.py` |
| **G8** | `20261010一楼大厅` 双角墙系平均成错误 24.5° 及斜墙外扩 ±1 m | 1D 高斯平滑加权众数主轴投票 + `≤ 5.5° / ≤ 0.16 m` 正交吸附 + 共线斜墙合并 + `abs(det) ≥ 0.45` 防尖刺交点与 `≤ 16 cm` 顶点贴云守卫（`frame_floors.py`） | `invariants_audit.max_footprint_vertex_cloud_dist_m` + `test_dual_angle_trapezoid_...` |
| **G9** | 大厅 1F 顶板中央 AI 误标类 1 挖大洞 + 2F 楼梯间顶板/高墙/高梁缺失 + `DCJS` 室外 `+4.38 m` 屋顶误入 | `WL-A2` 共面无上层房间时 `1→11`；高顶板须满足室内轮廓内占比 `≥ 55%`（放行大厅 `+2.08 m`，拦死 `DCJS +4.38 m`）；楼梯间外墙按实测高点分段拔高；梁/柱/扶手按局部顶板标高上限（`working_labels.py`, `frame_floors.py`, `walls_shell.py`, `beams_columns.py`） | `test_wl_a2_...` + `test_dual_angle_trapezoid_...` + `audit_scenes.py` |
| **G10** | 三跑 U 形楼梯扭曲压扶手、梯底 1F 柱阻挡二跑 | 递归多跑转折拆段 `_split_stair_turns_recursive` + `8°` 轴向吸附 + Z 感知梯底柱避让 + 贪心外部面积内接（`stairs_railings.py`） | `test_dual_angle_trapezoid_...` + `audit_scenes.py` |
| **G11** | 楼梯间缺地角落被切 + 上部过梁内墙下拉封死通道 + 楼梯/扶手越出类 13/14 边界 | 墙围合空腔恢复；内墙 2D `(s, z)` 侧立面异形/悬挑轮廓挤出；楼梯/平台硬交类 13 点云占用域（`+8 cm`），扶手严格贴附实测 `w_meas` 与分箱占用率 `≥ 55%`（`frame_floors.py`, `walls_shell.py`, `stairs_railings.py`） | `test_overhead_header_...` + `audit_scenes.py` |
| **G12** | **回归教训**：`TJHerlou` 1F 立柱挖穿蓝顶板 3 个洞 + 大厅楼梯转角平台被标成类 1 造成 30 cm 断层 | ① `ceiling_primary` 仅扣除**真正贯穿 `[zc_bot+1.5cm, zc_top-1.5cm]`** 的体块，严禁扣除下方 1F 柱/内墙，单层房间硬要求 `ceiling_primary_hole_count == 0`；② **`WL-D` 点云层先将夹在梯段间的紧凑平平台（`0.4–8.0 m²`）由 `1→13` 并入楼梯**，再由合并点云无缝重建 3 跑 + 2 完整转角平台（`working_labels.py`, `frame_floors.py`, `stairs_railings.py`, `pipeline.py`） | `invariants_audit.ceiling_primary_hole_count == 0` + `test_wl_d_...` + `test_single_storey_ceiling_sealed` + `audit_scenes.py` |
