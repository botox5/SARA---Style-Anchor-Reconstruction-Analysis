# SARA - Style Anchor Reconstruction & Analysis
## 基于风格锚点重构的作者身份验证系统

![SARA](https://img.shields.io/badge/version-1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)

---

## 📋 项目概述

**SARA** 是一个基于深度学习与统计方法的**作者身份验证系统**。它通过以下创新方法识别文本是否属于特定作者：

1. **特征锚点重构** - 从待测文本中提取和重构"风格锚点"（a）
2. **多维度对比** - 使用 6 种维度（虚词、标点、句法、语义、n-gram、情感）进行对比
3. **加权融合判定** - 综合多维特征，输出置信度与专家级建议

### 核心创新点

✨ **Burrows' Delta + AI 辅助**：结合经典作者验证方法与大模型风格分析  
✨ **端到端 PDF 处理**：自动提取 PDF 中的正文，智能跳过图表/表格/参考文献  
✨ **可视化报告**：生成交互式 HTML 报告，包含图表、表格、建议  
✨ **高度可配置**：权重、阈值、特征集合均可自定义  

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用 requirements.txt
pip install -r requirements.txt

# 或单独安装
pip install pdfplumber jieba sentence-transformers openai jinja2 numpy pandas matplotlib
```

### 2. 配置 API

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

### 3. 基础使用

```bash
# 语法
python SARA_complete_system.py <A语料PDF> <A'语料PDF>

# 示例
python SARA_complete_system.py paper_a.pdf paper_a_prime.pdf
```

### 4. 查看报告

报告自动生成在 `./sara_reports/SARA_Report_*.html`  
用任何浏览器打开即可。

---

## 📁 项目结构

```
SARA/
├── SARA_complete_system.py      # ⭐ 主系统代码（2500+ 行）
├── test_sara.py                 # 🧪 快速测试脚本
├── SARA_Guide_CN.md             # 📖 详细部署指南（中文）
├── README.md                    # 📝 本文件
├── requirements.txt             # 📦 依赖清单
├── sara_reports/                # 📊 输出报告目录（自动创建）
│   └── SARA_Report_*.html
├── sample_pdfs/                 # 📄 示例 PDF（可选）
│   ├── sample_text_a.pdf
│   └── sample_text_a_prime.pdf
└── data/                        # 💾 用户数据（可选）
    ├── your_paper_a.pdf
    └── your_paper_a_prime.pdf
```

---

## 🧠 工作原理

### 整体流程

```
输入：PDF 文件 A（待验证）和 A'（作者样本）
  ↓
[1] PDF 解析 → 纯文本提取 → 清理
  ↓
[2] 特征提取 → 虚词、标点、句法、语义特征
  ↓
[3] 风格描述 → 用 GPT-4 分析 A 的风格
  ↓
[4] 锚点生成 → 用 GPT-4 改写约 500 字的 a（保留风格）
  ↓
[5] 多维对比 → 计算 6 个相似度维度
  ↓
[6] 综合判定 → 加权融合 → 输出置信度
  ↓
[7] 报告生成 → 交互式 HTML 含图表与建议
  ↓
输出：SARA 报告 + 判定结果
```

### 核心算法

#### 1. **Burrows' Delta**（虚词指纹）
基于最高频 30-50 个虚词的频率分布，计算两段文本的距离。
- **公式**：$\Delta = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (Z_a(w_i) - Z_b(w_i))^2}$
- **优势**：难以伪造，是作者验证的金标准

#### 2. **标点相似度**
比较句号、逗号、感叹号等标点的分布，计算余弦相似度。
- **捕捉**：作者的"呼吸感"和节奏感

#### 3. **句法特征**
平均句长（ASL）、平均词长（AWL）、词汇丰富度（TTR）
- **捕捉**：作者的"思维速度"和"词汇品味"

#### 4. **语义相似度**
使用预训练多语言 BERT 模型的向量表示，计算余弦距离。
- **捕捉**：深层语义和逻辑结构

#### 5. **n-gram 相似度**
字符级 2-gram 的 Jaccard 相似度
- **捕捉**：字符序列的习惯

#### 6. **虚词使用习惯**
直接比较"但是"、"然而"、"所以"等虚词的频率
- **捕捉**：有意识的论证策略

### 权重配置（默认）

| 维度 | 权重 | 说明 |
|------|------|------|
| Burrows' Delta | 40% | 最重要，虚词指纹 |
| 虚词习惯相似度 | 20% | 论证策略 |
| 标点相似度 | 15% | 节奏感 |
| 句法相似度 | 15% | 思维速度 |
| 语义相似度 | 5% | 深层逻辑 |
| n-gram 相似度 | 5% | 字符习惯 |

---

## 📊 报告示例

SARA 生成的报告包含：

### 主要部分

1. **综合判定** - 最终结论（Match / Likely Match / Uncertain / Likely Mismatch / Mismatch）
2. **置信度展示** - 用色阶条形图展示（0-100%）
3. **多维度对比** - 6 个相似度指标的详细数据与可视化
4. **文本特征表** - 并排对比 a 与 A' 的所有统计特征
5. **虚词频率对比** - 前 10 个虚词的柱状图
6. **标点分布对比** - 各类标点使用情况
7. **风格描述** - AI 提取的原文本风格特征
8. **风格锚点** - AI 生成的约 500 字重构文本
9. **雷达图** - 多维相似度的直观展示
10. **专家建议** - 根据置信度给出后续建议

---

## 🔧 配置与自定义

### 修改权重

编辑 `SARA_complete_system.py` 中的 `VerdictEngine.compute_confidence()` 方法：

```python
confidence = (
    burrows_similarity * 0.40 +      # 修改这个值
    punctuation_sim * 0.15 +
    function_words_sim * 0.20 +
    syntactic_sim * 0.15 +
    semantic_sim * 0.05 +
    ngram_sim * 0.05
)
```

### 修改判定阈值

```python
if confidence > 0.80:           # 修改阈值
    verdict = "Match (强匹配)"
elif confidence > 0.60:
    verdict = "Likely Match (可能匹配)"
# ...
```

### 使用不同的 LLM 模型

```python
MODEL_NAME = "gpt-3.5-turbo"    # 更便宜但稍慢
MODEL_NAME = "gpt-4-turbo"      # 更准确但更贵
```

或使用开源模型（如 ollama + mistral）：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

### 自定义虚词列表

```python
FUNCTION_WORDS_ZH = [
    "你的虚词1", "你的虚词2", ...
]
```

---

## 🧪 测试

### 快速测试（无需 PDF）

```bash
# 测试所有功能
python test_sara.py all

# 仅测试特征提取
python test_sara.py features

# 虚词指纹演示
python test_sara.py function_words

# 性能压力测试
python test_sara.py stress

# 生成示例 PDF
python test_sara.py pdf
```

---

## 📈 性能指标

| 操作 | 耗时 | 备注 |
|------|------|------|
| 单 PDF 提取 | 1-3 秒 | 取决于文件大小 |
| 特征提取 | 0.5-1 秒 | ~1000字/秒 |
| API 调用（风格描述） | 3-5 秒 | 调用 GPT-4 |
| API 调用（锚点生成） | 5-8 秒 | 调用 GPT-4，生成 500 字 |
| 多维度对比 | <1 秒 | 本地计算 |
| 报告生成 | 1-2 秒 | 生成 HTML + 图表 |
| **总耗时** | **12-20 秒** | 端到端，含所有步骤 |

---

## ⚠️ 限制与已知问题

### 限制

1. **语言限制** - 目前仅支持中文，英文支持计划中
2. **文体限制** - 对学术论文效果最佳，创意写作（小说、诗歌）准确度较低
3. **样本量** - 建议 A' 至少 500 字以上，最佳 2000+ 字
4. **改写敏感度** - 若文本经历过大幅改写或他人编辑，准确度会下降
5. **API 成本** - 每次分析约消耗 $0.02-0.05 的 API 费用

### 已知问题

- PDF 扫描版（图片型）需要 OCR，暂不支持
- 某些特殊格式的 PDF（如权限保护）可能无法提取
- 中文分词有时不够准确，可通过调整 jieba 词典优化
- 首次运行需要下载 ~400MB 的句向量模型

---

## 🛠️ 故障排除

### PDF 提取失败

```python
# 测试 PDF 可读性
import pdfplumber
with pdfplumber.open("your_file.pdf") as pdf:
    print(pdf.pages[0].extract_text())
```

### API 超时

1. 检查网络连接
2. 检查 API 密钥有效性
3. 检查账户余额
4. 尝试降级模型（gpt-3.5-turbo）

### 内存不足

处理超大 PDF（>50MB）时：
```python
text_a = self.pdf_extractor.extract_text_from_pdf(pdf_a_path)[:5000]  # 截断到 5000 字
```

### 模型下载缓慢

```bash
# 手动下载并指定路径
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
```

---

## 📖 深入阅读

### 学术背景

- **Burrows' Delta**：Burrows, J. F. (2002). Delta: a measure of stylistic difference and a guide to likely authorship. *Literary and Linguistic Computing*, 17(3), 267-287.

- **PAN Authorship Verification**：官方评测集合 https://pan.webis.de/

- **预训练语言模型**：Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.

### 中文资源

- 《计算机辅助的作者归因研究综述》
- 作者验证竞赛数据集（PAN 2020-2023）
- jieba 中文分词文档

---

## 💡 最佳实践

### 数据准备

1. ✅ **清理 PDF**：移除水印、页眉页脚（若干扰分词）
2. ✅ **确保正文**：A' 应为作者的真实样本，非摘要/拼接
3. ✅ **多样性**：若可能，提供多份 A'（来自不同时期/体裁）
4. ✅ **合法性**：确保拥有分析文本的权利

### 解释结果

- **Match > 80%** → 极高置信度，可直接接受
- **Likely Match 60-80%** → 可能匹配，建议补充样本
- **Uncertain 45-60%** → 建议人工复审或获取更多证据
- **Mismatch < 45%** → 大概率不匹配，需要重新验证数据

### 注意事项

⚠️ **本系统仅供参考**，最终判定应结合人工专家评审。  
⚠️ **API 成本**：注意使用量，避免意外费用。  
⚠️ **隐私保护**：不在公开网络上传敏感文本。  

---

## 🤝 贡献与反馈

欢迎提交 Issue 和 Pull Request！

- 🐛 **Bug 报告**：请详细描述复现步骤
- 💡 **功能建议**：描述使用场景和预期行为
- 📊 **改进意见**：如何优化算法或报告

---

## 📄 许可证

MIT License - 自由使用、修改、分发，但需保留原始许可声明。

---

## 👨‍💻 作者

SARA 开发团队  
2025 年 12 月

---

## 🔗 相关资源

- [OpenAI API 文档](https://platform.openai.com/docs)
- [pdfplumber 文档](https://github.com/jsvine/pdfplumber)
- [jieba 分词文档](https://github.com/fxsjy/jieba)
- [Sentence-Transformers](https://www.sbert.net/)
- [PAN 作者验证竞赛](https://pan.webis.de/)

---

**最后更新**：2025-12-30  
**当前版本**：SARA v1.0
