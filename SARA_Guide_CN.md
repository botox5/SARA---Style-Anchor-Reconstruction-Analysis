# SARA 系统 - 完整部署指南

## 1. 环境要求

- Python 3.8+
- pip 包管理器

## 2. 依赖安装

```bash
# 创建虚拟环境（推荐）
python -m venv sara_env
source sara_env/bin/activate  # Linux/Mac
# 或
sara_env\Scripts\activate  # Windows

# 安装所有依赖
pip install -r requirements.txt
```

### requirements.txt 内容

```
pdfplumber>=0.10.0
PyPDF2>=3.0.0
jieba>=0.42.1
sentence-transformers>=2.2.0
scikit-learn>=1.0.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.12.0
openai>=1.0.0
jinja2>=3.0.0
```

### 逐个安装（如果 requirements.txt 出问题）

```bash
pip install pdfplumber PyPDF2 jieba
pip install sentence-transformers scikit-learn
pip install numpy pandas matplotlib seaborn
pip install openai jinja2
```

## 3. API 配置

### OpenAI API

1. 获取 API 密钥：https://platform.openai.com/api-keys
2. 设置环境变量：

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-..."
```

**Windows (cmd):**
```cmd
set OPENAI_API_KEY=sk-...
```

或直接在 Python 中设置：
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

## 4. 快速使用

### 基础用法

```bash
python SARA_complete_system.py <A语料PDF路径> <A'语料PDF路径>
```

### 实际示例

```bash
python SARA_complete_system.py ./data/article_a.pdf ./data/article_a_prime.pdf
```

### 输出

- 在 `./sara_reports/` 目录下生成 HTML 报告
- 文件名格式：`SARA_Report_YYYYMMDD_HHMMSS.html`
- 可用任何现代浏览器打开报告

## 5. 工作流程说明

```
输入
  ↓
[1] PDF 文本提取 → 清理、去除页码、过滤图表表格
  ↓
[2] 特征提取 → 虚词频率、标点分布、句法特征、嵌入向量
  ↓
[3] 风格描述生成 → 用 GPT 分析 A 的风格特征
  ↓
[4] 风格锚点生成 → 用 GPT 根据风格特征改写约 500 字的 a
  ↓
[5] 多维度对比计算
    ├─ Burrows' Delta（虚词距离）
    ├─ 标点相似度
    ├─ 虚词使用习惯
    ├─ 句法相似度（句长、词长）
    ├─ 语义相似度（嵌入）
    └─ n-gram 相似度
  ↓
[6] 综合判定 → 加权融合，输出置信度与判定
  ↓
输出
  └─ HTML 可视化报告（含图表、对比表、建议）
```

## 6. 重要参数调整

### API 模型选择

编辑 `SARA_complete_system.py` 第 53 行：

```python
MODEL_NAME = "gpt-4-turbo"  # 改为 "gpt-3.5-turbo" 以降低成本
```

### 虚词列表自定义

如果在某种特定领域，可调整第 57-68 行的 `FUNCTION_WORDS_ZH`：

```python
FUNCTION_WORDS_ZH = [
    "你的虚词1", "你的虚词2", ...
]
```

### 权重调整

在 `VerdictEngine.compute_confidence()` 方法中调整权重（默认）：

```python
confidence = (
    burrows_similarity * 0.40 +      # 虚词指纹（40%）
    punctuation_sim * 0.15 +         # 标点（15%）
    function_words_sim * 0.20 +      # 虚词习惯（20%）
    syntactic_sim * 0.15 +           # 句法（15%）
    semantic_sim * 0.05 +            # 语义（5%）
    ngram_sim * 0.05                 # n-gram（5%）
)
```

### 判定阈值调整

在同一方法中，修改以下条件：

```python
if confidence > 0.80:           # 修改这些阈值
    verdict = "Match (强匹配)"
elif confidence > 0.60:
    verdict = "Likely Match (可能匹配)"
# ...
```

## 7. 输出报告说明

### 报告包含内容

1. **综合判定**
   - 最终结论（Match / Likely Match / Uncertain / Likely Mismatch / Mismatch）
   - 置信度百分比
   - 详细理由

2. **多维度相似度分析**
   - Burrows' Delta 距离
   - 标点使用相似度
   - 虚词使用习惯相似度
   - 句法相似度
   - 语义相似度
   - 字符 n-gram 相似度

3. **文本特征对比**
   - 字数、词数、句数对比
   - 平均句长、平均词长对比
   - 词汇丰富度（TTR）对比
   - 情感评分对比

4. **高频虚词对比**
   - 前 10 个虚词及其使用频率
   - 可视化柱状图

5. **标点分布对比**
   - 各类标点出现次数对比

6. **风格描述与锚点**
   - AI 提取的原文本风格描述
   - AI 生成的约 500 字风格锚点

7. **可视化分析**
   - 雷达图展示多维度相似度

8. **建议**
   - 根据置信度给出后续建议

## 8. 常见问题排查

### Q: "无法加载句向量模型"

**A:** 第一次运行时，模型需要下载（约 400 MB）。确保网络连接正常。可尝试：

```bash
# 预先下载模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')"
```

### Q: PDF 提取失败

**A:** 确保：
1. PDF 文件路径正确
2. PDF 不是图片扫描版（需要 OCR）
3. PDF 有读取权限
4. 尝试用 pdfplumber 直接测试：

```python
import pdfplumber
with pdfplumber.open("your_file.pdf") as pdf:
    print(pdf.pages[0].extract_text())
```

### Q: API 调用超时

**A:** 
1. 检查网络连接
2. 检查 API 密钥有效性
3. 检查账户余额
4. 尝试降级模型：`gpt-3.5-turbo` 速度更快

### Q: 报告生成位置在哪？

**A:** 在执行脚本的目录下，会创建 `./sara_reports/` 文件夹，报告在其中。

### Q: 可以离线使用吗？

**A:** 不完全可以。以下部分需要网络：
- OpenAI API 调用（风格描述与锚点生成）
- 句向量模型首次下载

但可修改为本地替代方案：
```python
# 用 spaCy 代替 sentence-transformers
# 用 GPT 本地化版本（如 ollama + mistral）
```

## 9. 进阶用法

### 自定义特征提取

创建 `custom_features.py`：

```python
from SARA_complete_system import FeatureExtractor, TextFeatures

class CustomFeatureExtractor(FeatureExtractor):
    def extract_features(self, text):
        features = super().extract_features(text)
        # 添加你的自定义特征
        features.custom_score = self._compute_custom_metric(text)
        return features
```

### 批量处理多个文件对

```python
from SARA_complete_system import SARAPipeline

pipeline = SARAPipeline()
files = [
    ("a1.pdf", "a1_prime.pdf"),
    ("a2.pdf", "a2_prime.pdf"),
    ("a3.pdf", "a3_prime.pdf"),
]

for pdf_a, pdf_ap in files:
    result = pipeline.run_analysis(pdf_a, pdf_ap)
    print(f"{pdf_a}: {result.verdict} (信心度 {result.overall_confidence:.2%})")
```

### 修改报告模板

编辑 `ReportGenerator.generate_html_report()` 中的 `html_template` 部分。

## 10. 性能优化

### 加快处理速度

1. 使用 CPU 更快的嵌入模型：
   ```python
   self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
   ```

2. 缩短 PDF 文本（取前 5000 字）：
   ```python
   text_a = self.pdf_extractor.extract_text_from_pdf(pdf_a_path)[:5000]
   ```

3. 降低 API 调用复杂度（去掉风格描述生成）

## 11. 许可与引用

本系统基于以下学术工作：
- Burrows, J. F. (2002). "Delta": a measure of stylistic difference and a guide to likely authorship.
- PAN Authorship Verification shared task (2020-2023)

如在学术研究中使用，请引用本系统。

---

**更新时间**: 2025-12-30
**版本**: SARA v1.0
