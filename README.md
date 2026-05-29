# 水质监测与污染溯源助手

基于大语言模型 Agent 的水环境智能监测与溯源分析系统。

## 环境要求

| 要求 | 说明 |
|------|------|
| 操作系统 | Windows 10+（推荐）/ macOS / Linux |
| Python | **3.10 ~ 3.12**（推荐 3.11） |
| 网络 | 需能访问 `api.deepseek.com` |

## 快速启动（5 分钟）

### 1. 安装 Python

从 [python.org](https://www.python.org/downloads/) 下载安装 Python 3.11，**安装时务必勾选 "Add Python to PATH"**。

### 2. 启动项目

**Windows 用户**：双击 `启动.bat` 即可（自动安装依赖并启动）。

**macOS/Linux 用户**：在终端执行：

```bash
pip install --user -r requirements.txt
pip install --user -e water_quality_mcp/
streamlit run app.py
```

### 3. 配置 API Key

首次启动后，页面会弹出配置向导：

| 服务 | 用途 | 是否必需 | 获取地址 |
|------|------|---------|---------|
| **DeepSeek** | LLM 智能分析 | **必需** | [platform.deepseek.com](https://platform.deepseek.com) |
| 高德地图 | 地址反查、POI 搜索 | 可选 | [console.amap.com](https://console.amap.com) |
| 星图地球 | 卫星影像底图 | 可选 | [datacloud.geovisearth.com](https://datacloud.geovisearth.com) |

> 不填高德和星图不影响核心分析功能。

### 4. 加载数据并分析

1. 侧边栏点击「Browse files」上传走航数据（`苗圃杯/无人船智能水质监测分析和因果溯源样本数据/河道巡航走航样本数据.xlsx`）
2. （可选）上传样本视频（需等待上传完成再点击加载）
3. 点击「加载数据」
4. 点击「🚀 开始自动分析」
5. 切换到「📊 分析图表」标签查看结果

## 项目结构

```
├── app.py              # 主程序入口（Streamlit 前端）
├── build_animation.py  # 污染溯源动画生成器
├── 启动.bat             # Windows 一键启动脚本
├── requirements.txt    # 依赖列表（宽松版本）
├── requirements-lock.txt # 依赖列表（精确版本）
├── .env.example        # API Key 配置模板
├── .mcp.json.example   # MCP Server 配置模板
├── README.md           # 本文件
├── river_model/        # 一维水动力-水质模型（Preissmann + ADR）
├── water_quality_mcp/  # MCP Server + 分析工具库
└── 苗圃杯/             # 示例数据（Excel + 视频 + 参考文献）
```

## 常见问题

**Q: 双击启动.bat 闪退？**
确认已安装 Python 且勾选了 "Add Python to PATH"。在 cmd 中运行 `python --version` 验证。

**Q: 启动后显示"端口被占用"？**
脚本会自动尝试 8501 → 8502 → 8503，观察提示中的实际地址。

**Q: 时间序列图中文显示为方块？**
说明系统缺少中文字体。Windows 用户通常已预装，macOS/Linux 用户参考 `font_utils.py` 中的字体路径配置。

**Q: MCP Server 模式无法启动？**
需要先执行 `pip install -e water_quality_mcp/`，然后 `python -m water_quality_mcp.server`。

**Q: 视频上传后显示 0 帧？**
大视频文件上传需要时间，请等待文件名出现在上传框后再点击「加载数据」。
