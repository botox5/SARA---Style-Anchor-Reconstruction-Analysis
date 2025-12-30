#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SARA 系统 - 快速使用示例脚本

本脚本演示如何使用 SARA 系统进行作者身份验证分析。
可直接运行或作为模板修改后使用。
"""

import os
import sys
from pathlib import Path

# ==================================================================================
# 方案 1：直接使用命令行（推荐）
# ==================================================================================

def example_1_command_line():
    """
    最简单的使用方式 - 直接调用命令行
    
    优点：
    - 最快速
    - 无需编写代码
    - 自动处理所有流程
    
    缺点：
    - 不能自定义参数
    """
    
    print("""
    ╔═════════════════════════════════════════════════════════════════╗
    ║ 方案 1: 命令行直接使用                                         ║
    ╚═════════════════════════════════════════════════════════════════╝
    
    # 基础用法
    python SARA_complete_system.py paper_a.pdf paper_a_prime.pdf
    
    # 完整示例
    python SARA_complete_system.py \\
        /path/to/待验证文本.pdf \\
        /path/to/作者样本.pdf
    
    输出：HTML 报告在 ./sara_reports/SARA_Report_*.html
    """)

# ==================================================================================
# 方案 2：在 Python 脚本中调用
# ==================================================================================

def example_2_python_script():
    """
    在 Python 脚本中进行完全控制
    
    优点：
    - 可自定义各个步骤
    - 可批量处理多个文件对
    - 可修改算法参数
    
    缺点：
    - 需要编写代码
    - 需要理解系统架构
    """
    
    print("""
    ╔═════════════════════════════════════════════════════════════════╗
    ║ 方案 2: Python 脚本中使用                                      ║
    ╚═════════════════════════════════════════════════════════════════╝
    """)
    
    # 示例代码
    example_code = '''
from SARA_complete_system import SARAPipeline

# 1. 初始化 SARA pipeline
pipeline = SARAPipeline(api_key="sk-your-key")

# 2. 运行分析
result = pipeline.run_analysis(
    pdf_a_path="待验证文本.pdf",
    pdf_a_prime_path="作者样本.pdf"
)

# 3. 获取结果
print(f"置信度: {result.overall_confidence:.2%}")
print(f"判定: {result.verdict}")
print(f"理由: {result.reasoning['reason']}")

# 4. 访问各个相似度指标
print(f"虚词距离 (Burrows Delta): {result.burrows_delta:.4f}")
print(f"标点相似度: {result.punctuation_similarity:.2%}")
print(f"句法相似度: {result.syntactic_distance:.2%}")
print(f"语义相似度: {result.semantic_similarity:.2%}")
    '''
    
    print(example_code)

# ==================================================================================
# 方案 3：批量处理
# ==================================================================================

def example_3_batch_processing():
    """
    批量处理多个文本对
    """
    
    print("""
    ╔═════════════════════════════════════════════════════════════════╗
    ║ 方案 3: 批量处理多个文本对                                     ║
    ╚═════════════════════════════════════════════════════════════════╝
    """)
    
    example_code = '''
from SARA_complete_system import SARAPipeline
import json

# 初始化
pipeline = SARAPipeline()

# 定义文件对列表
file_pairs = [
    ("paper1_a.pdf", "paper1_a_prime.pdf"),
    ("paper2_a.pdf", "paper2_a_prime.pdf"),
    ("paper3_a.pdf", "paper3_a_prime.pdf"),
]

# 批量分析
results = {}
for pdf_a, pdf_ap in file_pairs:
    try:
        result = pipeline.run_analysis(pdf_a, pdf_ap)
        results[pdf_a] = {
            "confidence": f"{result.overall_confidence:.2%}",
            "verdict": result.verdict
        }
        print(f"✓ {pdf_a}: {result.verdict}")
    except Exception as e:
        results[pdf_a] = {"error": str(e)}
        print(f"✗ {pdf_a}: {e}")

# 保存结果到 JSON
with open("analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
    ''')
    
    print(example_code)

# ==================================================================================
# 方案 4：自定义参数
# ==================================================================================

def example_4_custom_parameters():
    """
    修改算法参数以适应特定需求
    """
    
    print("""
    ╔═════════════════════════════════════════════════════════════════╗
    ║ 方案 4: 自定义参数                                             ║
    ╚═════════════════════════════════════════════════════════════════╝
    """)
    
    example_code = '''
from SARA_complete_system import (
    StyleComparator, VerdictEngine, FeatureExtractor
)

# 4.1 修改权重
class CustomVerdictEngine(VerdictEngine):
    @staticmethod
    def compute_confidence(
        burrows_delta, punctuation_sim, semantic_sim, 
        ngram_sim, function_words_sim, syntactic_sim
    ):
        # 自定义权重：更强调虚词
        burrows_similarity = 1 - min(burrows_delta, 1.0)
        confidence = (
            burrows_similarity * 0.50 +      # 从 40% 增加到 50%
            punctuation_sim * 0.10 +         # 从 15% 降低到 10%
            function_words_sim * 0.20 +
            syntactic_sim * 0.10 +           # 从 15% 降低到 10%
            semantic_sim * 0.05 +
            ngram_sim * 0.05
        )
        # ... 返回 confidence, verdict, reasoning
        return confidence, verdict, reasoning

# 4.2 修改判定阈值
def custom_verdict(confidence):
    if confidence > 0.85:           # 更严格
        return "Strong Match"
    elif confidence > 0.70:
        return "Probable Match"
    elif confidence > 0.50:
        return "Possible Match"
    else:
        return "Not a Match"

# 4.3 使用自定义的特征提取器
class CustomFeatureExtractor(FeatureExtractor):
    def extract_features(self, text):
        features = super().extract_features(text)
        # 添加自定义特征
        features.custom_metric = self._compute_custom_metric(text)
        return features
    
    def _compute_custom_metric(self, text):
        # 你的自定义特征计算逻辑
        return 0.5
    ''')
    
    print(example_code)

# ==================================================================================
# 方案 5：集成到现有系统
# ==================================================================================

def example_5_integration():
    """
    将 SARA 集成到现有的文档管理或内容审核系统
    """
    
    print("""
    ╔═════════════════════════════════════════════════════════════════╗
    ║ 方案 5: 集成到现有系统                                         ║
    ╚═════════════════════════════════════════════════════════════════╝
    """)
    
    example_code = '''
from SARA_complete_system import SARAPipeline
from flask import Flask, request, jsonify
import tempfile
import os

app = Flask(__name__)
pipeline = SARAPipeline()

@app.route('/verify_authorship', methods=['POST'])
def verify_authorship():
    """
    API 端点：验证文本作者身份
    
    请求格式：
    {
        "file_a": <二进制 PDF 或 Base64>,
        "file_a_prime": <二进制 PDF 或 Base64>
    }
    
    响应格式：
    {
        "confidence": 0.85,
        "verdict": "Match (强匹配)",
        "details": {...}
    }
    """
    
    try:
        # 获取上传的文件
        file_a = request.files['file_a']
        file_a_prime = request.files['file_a_prime']
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_a:
            file_a.save(tmp_a.name)
            pdf_a_path = tmp_a.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_ap:
            file_a_prime.save(tmp_ap.name)
            pdf_a_prime_path = tmp_ap.name
        
        # 运行分析
        result = pipeline.run_analysis(pdf_a_path, pdf_a_prime_path)
        
        # 清理临时文件
        os.unlink(pdf_a_path)
        os.unlink(pdf_a_prime_path)
        
        # 返回结果
        return jsonify({
            "confidence": result.overall_confidence,
            "verdict": result.verdict,
            "reasoning": result.reasoning,
            "details": {
                "burrows_delta": result.burrows_delta,
                "punctuation_similarity": result.punctuation_similarity,
                "semantic_similarity": result.semantic_similarity,
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, port=5000)
    
# 使用示例：
# curl -X POST http://localhost:5000/verify_authorship \\
#   -F "file_a=@paper_a.pdf" \\
#   -F "file_a_prime=@paper_a_prime.pdf"
    ''')
    
    print(example_code)

# ==================================================================================
# 方案 6：命令行工具包装
# ==================================================================================

def example_6_cli_wrapper():
    """
    创建更友好的命令行界面
    """
    
    print("""
    ╔═════════════════════════════════════════════════════════════════╗
    ║ 方案 6: 高级命令行工具                                         ║
    ╚═════════════════════════════════════════════════════════════════╝
    """)
    
    example_code = '''
import click
from SARA_complete_system import SARAPipeline
from pathlib import Path
import json

@click.group()
def cli():
    """SARA - 作者身份验证系统"""
    pass

@cli.command()
@click.argument('file_a', type=click.Path(exists=True))
@click.argument('file_a_prime', type=click.Path(exists=True))
@click.option('--output', '-o', default='sara_reports', help='输出目录')
@click.option('--threshold', '-t', default=0.8, type=float, help='判定阈值')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
def analyze(file_a, file_a_prime, output, threshold, verbose):
    """分析两个 PDF 文件的作者相似度"""
    
    click.echo("🚀 SARA 分析开始...")
    
    pipeline = SARAPipeline()
    result = pipeline.run_analysis(file_a, file_a_prime)
    
    click.echo(f"\\n置信度: {result.overall_confidence:.2%}")
    click.echo(f"判定: {result.verdict}")
    
    if verbose:
        click.echo("\\n【详细指标】")
        click.echo(f"  Burrows Delta: {result.burrows_delta:.4f}")
        click.echo(f"  标点相似度: {result.punctuation_similarity:.2%}")
        click.echo(f"  句法相似度: {result.syntactic_distance:.2%}")
        click.echo(f"  语义相似度: {result.semantic_similarity:.2%}")
    
    click.echo(f"\\n✅ 报告已生成")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
def batch(input_file):
    """批量分析 JSON 配置文件中的文本对"""
    
    with open(input_file, 'r') as f:
        pairs = json.load(f)
    
    pipeline = SARAPipeline()
    
    for pair in pairs:
        click.echo(f"处理: {pair['name']}...")
        result = pipeline.run_analysis(pair['file_a'], pair['file_a_prime'])
        click.echo(f"  结果: {result.verdict} ({result.overall_confidence:.2%})")

if __name__ == '__main__':
    cli()
    
# 使用示例：
# python sara_cli.py analyze paper_a.pdf paper_a_prime.pdf --verbose
# python sara_cli.py batch config.json
    ''')
    
    print(example_code)

# ==================================================================================
# 主函数
# ==================================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║         SARA 系统 - 6 种使用方式快速参考指南                      ║
    ║     Style Anchor Reconstruction & Analysis v1.0                   ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        method = sys.argv[1]
        if method == "1":
            example_1_command_line()
        elif method == "2":
            example_2_python_script()
        elif method == "3":
            example_3_batch_processing()
        elif method == "4":
            example_4_custom_parameters()
        elif method == "5":
            example_5_integration()
        elif method == "6":
            example_6_cli_wrapper()
        else:
            print(f"未知选项: {method}\\n请选择 1-6")
    else:
        # 显示所有方案摘要
        print("""
┌────────────────────────────────────────────────────────────────────┐
│ 方案 1: 命令行直接使用 (最快)                                    │
│ 适用：初次使用、一次性分析                                        │
│ 命令：python SARA_complete_system.py <A> <A'>                    │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ 方案 2: Python 脚本 (推荐)                                       │
│ 适用：需要访问详细结果、集成到项目中                              │
│ 优点：完全控制、易于扩展                                          │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ 方案 3: 批量处理 (高效)                                          │
│ 适用：大量文件分析、批量验证                                      │
│ 优点：自动化、可生成统计报告                                      │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ 方案 4: 自定义参数 (灵活)                                        │
│ 适用：特殊领域、调参优化                                          │
│ 优点：精细控制算法权重和阈值                                      │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ 方案 5: API 集成 (强大)                                          │
│ 适用：Web 服务、企业应用                                          │
│ 优点：RESTful 接口、易于部署                                      │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ 方案 6: CLI 工具 (专业)                                          │
│ 适用：命令行用户、CI/CD 流程                                      │
│ 优点：简洁、强大、可扩展                                          │
└────────────────────────────────────────────────────────────────────┘

查看具体示例：
  python example_usage.py 1      # 查看方案 1
  python example_usage.py 2      # 查看方案 2
  python example_usage.py 3      # 查看方案 3
  python example_usage.py 4      # 查看方案 4
  python example_usage.py 5      # 查看方案 5
  python example_usage.py 6      # 查看方案 6

更多信息请参考：
  - README_CN.md (项目说明)
  - SARA_Guide_CN.md (部署指南)
  - SARA_complete_system.py (源代码注释)
        """)
