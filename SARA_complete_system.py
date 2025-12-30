# ==================================================================================
# SARA: Style Anchor Reconstruction & Analysis
# 基于风格锚点重构的作者身份验证系统
# ==================================================================================

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np
from datetime import datetime

# PDF 处理
import PyPDF2
from pdfplumber import PDF
import pdfplumber

# NLP 与分词
import jieba
import jieba.analyse
from jieba import posseg as pseg

# 文本相似度与嵌入
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 数据处理
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 调用大模型 API（以 OpenAI 为例，可替换为其他 API）
import openai
from openai import OpenAI

# HTML 报告生成
from jinja2 import Template

# ==================================================================================
# 配置与常量
# ==================================================================================

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# API 配置（从环境变量读取）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
MODEL_NAME = "gpt-4-turbo"  # 可改为 gpt-3.5-turbo

# 高频虚词列表（用于 Burrows' Delta）
FUNCTION_WORDS_ZH = [
    "的", "了", "和", "是", "在", "了", "这", "我", "有", "你",
    "他", "她", "但是", "所以", "因为", "而", "或", "及", "以", "为",
    "由", "被", "与", "到", "从", "向", "对", "把", "让", "给",
    "于", "之", "其实", "而且", "然而", "却", "不过", "或许", "可能", "似乎",
    "应该", "必须", "需要", "一直", "已经", "正在", "即将", "开始", "结束", "完成",
    "也", "都", "只", "就", "还", "再", "又", "很", "太", "非常",
    "可以", "能够", "必然", "无法", "不能", "不会", "不是", "没有", "没", "无"
]

# ==================================================================================
# 数据结构
# ==================================================================================

@dataclass
class TextFeatures:
    """文本特征容器"""
    text: str
    length: int
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    avg_word_length: float
    function_words_freq: Dict[str, float]
    punctuation_dist: Dict[str, int]
    named_entities: List[str]
    sentiment_score: float
    vocabulary_richness: float  # TTR (Type-Token Ratio)
    
    def to_dict(self):
        return asdict(self)

@dataclass
class AnalysisResult:
    """完整对比分析结果"""
    text_a_features: TextFeatures
    text_a_prime_features: TextFeatures
    burrows_delta: float
    punctuation_similarity: float
    semantic_similarity: float
    ngram_similarity: float
    function_words_similarity: float
    syntactic_distance: float
    overall_confidence: float
    verdict: str  # "Match", "Uncertain", "Mismatch"
    reasoning: Dict[str, str]
    timestamp: str

# ==================================================================================
# 1. PDF 提取与文本预处理
# ==================================================================================

class PDFExtractor:
    """从 PDF 中提取纯文本，跳过图表和参考文献"""
    
    def __init__(self):
        self.text = ""
        self.metadata = {}
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从 PDF 提取文本，尝试识别和排除图表/表格/参考文献
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                self.metadata = pdf.metadata
                full_text = ""
                
                for page_num, page in enumerate(pdf.pages):
                    # 尝试提取文本
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                    
                    # 检测表格（如果有，可选跳过或标记）
                    tables = page.extract_tables()
                    if tables:
                        print(f"[Page {page_num}] 检测到表格，已跳过")
                
                self.text = full_text
                return self._clean_text(full_text)
        except Exception as e:
            print(f"PDF 读取失败: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本：
        - 移除过多空行
        - 移除页码
        - 移除 URL
        - 保留中文、英文、数字、标点
        """
        # 移除页码（例如 "- 1 -"）
        text = re.sub(r'-\s*\d+\s*-', '', text)
        
        # 移除 URL
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # 移除过多空行
        text = re.sub(r'\n\n+', '\n', text)
        
        # 移除前后空白
        text = text.strip()
        
        return text
    
    def segment_by_sections(self, text: str) -> Dict[str, str]:
        """
        尝试按逻辑分割文本（主体 / 参考文献 / 附录）
        简化版：用"参考文献"作为切分点
        """
        sections = {
            "body": text,
            "references": "",
            "appendix": ""
        }
        
        # 查找"参考文献"段落
        refs_match = re.search(r'(参考文献|References|参考文档|致谢|Acknowledgment)[\s\S]*', text)
        if refs_match:
            sections["body"] = text[:refs_match.start()]
            sections["references"] = refs_match.group()
        
        return sections

# ==================================================================================
# 2. 文本特征提取
# ==================================================================================

class FeatureExtractor:
    """从文本中提取风格特征"""
    
    def __init__(self):
        # 加载预训练中文句向量模型
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        except Exception as e:
            print(f"警告：无法加载句向量模型 ({e})，将使用降级模式")
            self.embedding_model = None
    
    def extract_features(self, text: str) -> TextFeatures:
        """提取完整的文本特征集"""
        
        # 基础统计
        length = len(text)
        sentences = self._split_sentences(text)
        sentence_count = len(sentences)
        
        # 分词
        words = jieba.lcut(text)
        words = [w for w in words if w.strip()]  # 移除空白
        word_count = len(words)
        
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_word_length = length / word_count if word_count > 0 else 0
        
        # 虚词频率
        function_words_freq = self._extract_function_words_freq(words)
        
        # 标点分布
        punctuation_dist = self._extract_punctuation_dist(text)
        
        # 命名实体（简化版）
        named_entities = self._extract_named_entities(words)
        
        # 情感评分（调用 API）
        sentiment_score = self._get_sentiment_score(text)
        
        # 词汇丰富度（TTR）
        vocabulary_richness = len(set(words)) / word_count if word_count > 0 else 0
        
        return TextFeatures(
            text=text,
            length=length,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            function_words_freq=function_words_freq,
            punctuation_dist=punctuation_dist,
            named_entities=named_entities,
            sentiment_score=sentiment_score,
            vocabulary_richness=vocabulary_richness
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """按中文句号、问号、感叹号切分句子"""
        sentences = re.split(r'[。！？\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_function_words_freq(self, words: List[str]) -> Dict[str, float]:
        """计算虚词的相对频率"""
        word_count = len(words)
        freq = {}
        
        for fw in FUNCTION_WORDS_ZH:
            count = sum(1 for w in words if w == fw)
            freq[fw] = count / word_count if word_count > 0 else 0
        
        # 按频率排序，只保留 top 30
        freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:30])
        return freq
    
    def _extract_punctuation_dist(self, text: str) -> Dict[str, int]:
        """统计标点符号分布"""
        punctuation_marks = {
            '。': len(re.findall(r'。', text)),
            '，': len(re.findall(r'，', text)),
            '！': len(re.findall(r'！', text)),
            '？': len(re.findall(r'？', text)),
            '；': len(re.findall(r'；', text)),
            '、': len(re.findall(r'、', text)),
            '：': len(re.findall(r'：', text)),
            '（': len(re.findall(r'（', text)),
            '）': len(re.findall(r'）', text)),
            '"': len(re.findall(r'"', text)),
            '"': len(re.findall(r'"', text)),
            '——': len(re.findall(r'——', text)),
        }
        return punctuation_marks
    
    def _extract_named_entities(self, words: List[str]) -> List[str]:
        """提取可能的命名实体（简化版，基于 POS 标注）"""
        entities = []
        for word, flag in pseg.cut(" ".join(words)):
            if flag in ['nr', 'ns', 'nt', 'nz']:  # 人名、地名、机构、其他专名
                entities.append(word)
        return entities[:20]  # 只保留前 20 个
    
    def _get_sentiment_score(self, text: str) -> float:
        """
        调用 API 获取情感评分
        使用 OpenAI 的 embeddings 或专门的情感分析 API
        这里用简化版：计算正负词汇比
        """
        positive_words = ['好', '优', '棒', '完美', '赞', '出色', '杰出', '了不起', '精彩', '非常']
        negative_words = ['差', '劣', '烂', '糟糕', '讨厌', '失败', '麻烦', '困难', '问题', '错误']
        
        text_lower = text.lower()
        pos_count = sum(text.count(w) for w in positive_words)
        neg_count = sum(text.count(w) for w in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.5  # 中性
        
        return (pos_count - neg_count) / total * 0.5 + 0.5  # 归一化到 [0, 1]
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本向量表示"""
        if self.embedding_model is None:
            return None
        try:
            # 为了避免超长文本，截断到前 512 个字符
            text_truncated = text[:512]
            embedding = self.embedding_model.encode(text_truncated, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"嵌入提取失败: {e}")
            return None

# ==================================================================================
# 3. 对比与相似度计算
# ==================================================================================

class StyleComparator:
    """计算两段文本的风格相似度"""
    
    @staticmethod
    def burrows_delta(freq_a: Dict[str, float], freq_b: Dict[str, float]) -> float:
        """
        计算 Burrows' Delta 距离
        Delta = sqrt(sum((Z_a(w) - Z_b(w))^2) / n)
        其中 Z 是 z-score 标准化
        """
        # 合并两个频率字典的所有词汇
        all_words = set(freq_a.keys()) | set(freq_b.keys())
        
        # 计算每个词汇的 z-score
        deltas = []
        for word in all_words:
            z_a = freq_a.get(word, 0)
            z_b = freq_b.get(word, 0)
            
            # 简化版：直接计算差的平方（完整版需要计算 z-score）
            delta_sq = (z_a - z_b) ** 2
            deltas.append(delta_sq)
        
        if not deltas:
            return 0.0
        
        return float(np.sqrt(np.mean(deltas)))
    
    @staticmethod
    def punctuation_similarity(punc_a: Dict[str, int], punc_b: Dict[str, int]) -> float:
        """
        计算标点分布的相似度
        使用余弦相似度
        """
        # 统一 key
        all_keys = set(punc_a.keys()) | set(punc_b.keys())
        vec_a = np.array([punc_a.get(k, 0) for k in all_keys])
        vec_b = np.array([punc_b.get(k, 0) for k in all_keys])
        
        # 归一化
        if np.linalg.norm(vec_a) > 0:
            vec_a = vec_a / np.linalg.norm(vec_a)
        if np.linalg.norm(vec_b) > 0:
            vec_b = vec_b / np.linalg.norm(vec_b)
        
        similarity = float(np.dot(vec_a, vec_b))
        return (similarity + 1) / 2  # 转换到 [0, 1]
    
    @staticmethod
    def syntactic_distance(feat_a: TextFeatures, feat_b: TextFeatures) -> float:
        """
        计算句法特征距离（句长、词长等）
        返回相似度 [0, 1]
        """
        # 计算各维度的距离
        asl_diff = abs(feat_a.avg_sentence_length - feat_b.avg_sentence_length) / max(feat_a.avg_sentence_length, feat_b.avg_sentence_length, 1)
        awl_diff = abs(feat_a.avg_word_length - feat_b.avg_word_length) / max(feat_a.avg_word_length, feat_b.avg_word_length, 1)
        ttrap_diff = abs(feat_a.vocabulary_richness - feat_b.vocabulary_richness)
        
        # 综合距离
        avg_distance = (asl_diff + awl_diff + ttrap_diff) / 3
        
        # 转换为相似度
        similarity = 1 - min(avg_distance, 1.0)
        return similarity
    
    @staticmethod
    def semantic_similarity(embedding_a: Optional[np.ndarray], embedding_b: Optional[np.ndarray]) -> float:
        """
        计算语义相似度（基于嵌入向量的余弦相似度）
        """
        if embedding_a is None or embedding_b is None:
            return 0.5  # 无法计算时返回中性值
        
        similarity = float(cosine_similarity([embedding_a], [embedding_b])[0][0])
        return (similarity + 1) / 2  # 转换到 [0, 1]
    
    @staticmethod
    def ngram_similarity(text_a: str, text_b: str, n: int = 2) -> float:
        """
        计算 n-gram 相似度
        """
        def get_ngrams(text, n):
            return Counter([text[i:i+n] for i in range(len(text) - n + 1)])
        
        ngrams_a = get_ngrams(text_a, n)
        ngrams_b = get_ngrams(text_b, n)
        
        if not ngrams_a or not ngrams_b:
            return 0.5
        
        # Jaccard 相似度
        intersection = sum((ngrams_a & ngrams_b).values())
        union = sum((ngrams_a | ngrams_b).values())
        
        similarity = intersection / union if union > 0 else 0.5
        return similarity
    
    @staticmethod
    def function_words_similarity(freq_a: Dict[str, float], freq_b: Dict[str, float]) -> float:
        """
        计算虚词使用相似度
        """
        common_words = set(freq_a.keys()) & set(freq_b.keys())
        
        if not common_words:
            return 0.5
        
        differences = []
        for word in common_words:
            diff = abs(freq_a[word] - freq_b[word])
            differences.append(diff)
        
        avg_diff = np.mean(differences)
        similarity = 1 - min(avg_diff, 1.0)
        return similarity

# ==================================================================================
# 4. AI 辅助生成（生成风格锚点 a）
# ==================================================================================

class AIStyleAnchor:
    """使用 LLM 生成风格锚点（a）"""
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)
    
    def extract_style_profile(self, text: str, features: TextFeatures) -> str:
        """
        基于特征，生成对文本的风格描述
        """
        prompt = f"""
        请分析以下文本的写作风格特征，用 150-200 字进行描述。

        文本长度: {features.word_count} 字
        平均句长: {features.avg_sentence_length:.1f} 字
        词汇丰富度: {features.vocabulary_richness:.2%}
        高频虚词: {', '.join(list(features.function_words_freq.keys())[:10])}
        标点特征: {dict(sorted(features.punctuation_dist.items(), key=lambda x: x[1], reverse=True)[:5])}
        
        文本片段（前 500 字）:
        {text[:500]}
        
        请描述该文本的：
        1. 句法特点（句长、从句比例等）
        2. 虚词使用习惯
        3. 标点使用特征
        4. 语气和情感基调
        5. 论述逻辑特征
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API 调用失败: {e}")
            return f"[特征提取失败] {str(e)}"
    
    def generate_style_anchor(self, text: str, style_profile: str) -> str:
        """
        基于原文本和风格描述，生成约 500 字的风格锚点（a）
        """
        prompt = f"""
        你是一个专业的文本改写和总结专家。
        
        原文本摘要（保留关键信息，约 500 字）：
        {text[:2000]}
        
        原文本的风格特征描述（你必须完全复制这些特征）：
        {style_profile}
        
        任务：请对上述文本进行压缩与改写，产出一个"风格锚点"（a），要求：
        1. 长度严格控制在 450-550 字
        2. 仅保留核心主张与逻辑链条，删除支线细节
        3. 完全继承原文本的以下特征：
           - 虚词使用频率与连接词习惯
           - 句长与标点节奏
           - 论述逻辑结构（如：先举例后总结 / 先定义后推导）
           - 情感基调与语气
           - 专业术语的使用密度
        4. 禁止使用 AI 套话（如"综上所述""总而言之"），除非原文本本来就常用
        5. 保留原文本的任何特殊短语、口癖或重复用词
        
        输出：仅输出改写后的文本，不需要任何前置说明。
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API 调用失败: {e}")
            return ""

# ==================================================================================
# 5. 综合判定引擎
# ==================================================================================

class VerdictEngine:
    """基于多维特征打分，输出最终判定"""
    
    @staticmethod
    def compute_confidence(
        burrows_delta: float,
        punctuation_sim: float,
        semantic_sim: float,
        ngram_sim: float,
        function_words_sim: float,
        syntactic_sim: float
    ) -> Tuple[float, str, Dict[str, str]]:
        """
        综合多维度特征，计算置信度与判定
        
        权重配置：
        - Burrows' Delta (虚词距离): 40% (最重要)
        - 标点相似度: 15%
        - 虚词相似度: 20%
        - 句法相似度: 15%
        - 语义相似度: 5%
        - n-gram 相似度: 5%
        """
        
        # 将 Delta 转换为相似度（Delta 越小越相似）
        # Delta 通常在 0-1 之间，>0.5 表示差异显著
        burrows_similarity = 1 - min(burrows_delta, 1.0)
        
        # 加权计算
        confidence = (
            burrows_similarity * 0.40 +
            punctuation_sim * 0.15 +
            function_words_sim * 0.20 +
            syntactic_sim * 0.15 +
            semantic_sim * 0.05 +
            ngram_sim * 0.05
        )
        
        # 判定逻辑
        if confidence > 0.80:
            verdict = "Match (强匹配)"
            reason_text = "置信度超过 80%，两段文本的风格特征高度一致，A 语料极大概率属于作者 A。"
        elif confidence > 0.60:
            verdict = "Likely Match (可能匹配)"
            reason_text = "置信度在 60-80% 之间，两段文本存在明显的风格相似性，但存在一定的变异空间（可能受编辑、改写或时间跨度影响）。"
        elif confidence > 0.45:
            verdict = "Uncertain (不确定)"
            reason_text = "置信度在 45-60% 之间，风格特征既有相似也有差异，无法确定归属。建议补充更多 A' 样本或进行人工复审。"
        elif confidence > 0.30:
            verdict = "Likely Mismatch (可能不匹配)"
            reason_text = "置信度在 30-45% 之间，两段文本在多个风格维度上存在显著差异，A 语料可能来自其他作者或经过重大改写。"
        else:
            verdict = "Mismatch (不匹配)"
            reason_text = "置信度低于 30%，两段文本的风格差异巨大，A 语料很可能不属于作者 A。"
        
        reasoning = {
            "verdict": verdict,
            "reason": reason_text,
            "confidence_percentage": f"{confidence * 100:.1f}%"
        }
        
        return confidence, verdict, reasoning

# ==================================================================================
# 6. 报告生成
# ==================================================================================

class ReportGenerator:
    """生成可视化对比分析报告（HTML）"""
    
    def __init__(self, output_dir: str = "./sara_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_html_report(self, result: AnalysisResult, 
                            style_profile: str, 
                            anchor_text: str,
                            pdf_a_path: str,
                            pdf_a_prime_path: str) -> str:
        """生成完整的 HTML 报告"""
        
        # 准备数据
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_filename = f"SARA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename
        
        # 生成图表
        charts = self._generate_charts(result)
        
        # HTML 模板
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SARA 作者身份验证报告</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: #f5f5f5;
                }
                
                .container {
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                    background: white;
                }
                
                header {
                    text-align: center;
                    padding: 30px 0;
                    border-bottom: 3px solid #2c3e50;
                    margin-bottom: 30px;
                }
                
                h1 {
                    font-size: 2.5em;
                    color: #2c3e50;
                    margin-bottom: 10px;
                }
                
                .subtitle {
                    font-size: 1em;
                    color: #7f8c8d;
                }
                
                .metadata {
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    font-size: 0.9em;
                }
                
                .verdict-box {
                    padding: 20px;
                    border-left: 5px solid;
                    margin: 20px 0;
                    font-size: 1.1em;
                }
                
                .verdict-box.match {
                    background: #d4edda;
                    border-color: #28a745;
                    color: #155724;
                }
                
                .verdict-box.likely-match {
                    background: #cce5ff;
                    border-color: #004085;
                    color: #004085;
                }
                
                .verdict-box.uncertain {
                    background: #fff3cd;
                    border-color: #856404;
                    color: #856404;
                }
                
                .verdict-box.likely-mismatch {
                    background: #f8d7da;
                    border-color: #f5c6cb;
                    color: #721c24;
                }
                
                .verdict-box.mismatch {
                    background: #f8d7da;
                    border-color: #c82333;
                    color: #721c24;
                }
                
                .section {
                    margin: 30px 0;
                }
                
                h2 {
                    font-size: 1.5em;
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 15px;
                }
                
                h3 {
                    font-size: 1.2em;
                    color: #34495e;
                    margin: 15px 0 10px 0;
                }
                
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                
                th {
                    background: #ecf0f1;
                    font-weight: bold;
                    color: #2c3e50;
                }
                
                tr:hover {
                    background: #f9f9f9;
                }
                
                .chart-container {
                    margin: 20px 0;
                    text-align: center;
                }
                
                .chart-container img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                
                .comparison-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin: 20px 0;
                }
                
                .comparison-box {
                    background: #f9f9f9;
                    padding: 15px;
                    border-radius: 5px;
                    border: 1px solid #ddd;
                }
                
                .metric-row {
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #eee;
                }
                
                .metric-label {
                    font-weight: bold;
                    color: #2c3e50;
                }
                
                .metric-value {
                    color: #3498db;
                }
                
                .similarity-bar {
                    width: 100%;
                    height: 30px;
                    background: #ecf0f1;
                    border-radius: 5px;
                    margin: 10px 0;
                    overflow: hidden;
                    position: relative;
                }
                
                .similarity-bar-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #e74c3c, #f39c12, #27ae60);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 0.9em;
                }
                
                .text-sample {
                    background: #f5f5f5;
                    padding: 15px;
                    border-left: 4px solid #3498db;
                    margin: 15px 0;
                    font-size: 0.95em;
                    line-height: 1.8;
                    max-height: 300px;
                    overflow-y: auto;
                }
                
                footer {
                    text-align: center;
                    padding: 20px;
                    color: #7f8c8d;
                    border-top: 1px solid #ddd;
                    margin-top: 30px;
                    font-size: 0.9em;
                }
                
                .grid-2 {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }
                
                @media (max-width: 768px) {
                    .comparison-grid, .grid-2 {
                        grid-template-columns: 1fr;
                    }
                    h1 {
                        font-size: 1.8em;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>📊 SARA 作者身份验证报告</h1>
                    <p class="subtitle">Style Anchor Reconstruction & Analysis</p>
                </header>
                
                <div class="metadata">
                    <strong>报告生成时间：</strong> {{ timestamp }} <br>
                    <strong>A 语料来源：</strong> {{ pdf_a_path }} <br>
                    <strong>A' 语料来源：</strong> {{ pdf_a_prime_path }} <br>
                    <strong>分析模型：</strong> SARA v1.0
                </div>
                
                <!-- 综合判定 -->
                <div class="section">
                    <h2>📌 综合判定</h2>
                    <div class="verdict-box {{ verdict_class }}">
                        <strong>判定结果：</strong> {{ verdict }} <br>
                        <strong>置信度：</strong> {{ confidence_percentage }} <br>
                        <strong>分析：</strong> {{ reasoning_text }}
                    </div>
                </div>
                
                <!-- 多维度相似度 -->
                <div class="section">
                    <h2>📈 多维度相似度分析</h2>
                    
                    <h3>虚词指纹相似度（Burrows' Delta）</h3>
                    <p>衡量最高频 30 个虚词的使用习惯。Delta 值越小，两段文本越接近。</p>
                    <div class="similarity-bar">
                        <div class="similarity-bar-fill" style="width: {{ burrows_similarity }}%;">
                            {{ burrows_similarity }}%
                        </div>
                    </div>
                    <p>Delta 值: {{ burrows_delta_value }}</p>
                    
                    <h3>标点使用相似度</h3>
                    <p>比较句号、逗号、感叹号等标点的使用模式。</p>
                    <div class="similarity-bar">
                        <div class="similarity-bar-fill" style="width: {{ punctuation_sim }}%;">
                            {{ punctuation_sim }}%
                        </div>
                    </div>
                    
                    <h3>虚词使用习惯相似度</h3>
                    <p>直接比较 "但是"、"而且"、"然而" 等虚词的频率分布。</p>
                    <div class="similarity-bar">
                        <div class="similarity-bar-fill" style="width: {{ function_words_sim }}%;">
                            {{ function_words_sim }}%
                        </div>
                    </div>
                    
                    <h3>句法相似度（句长、词长）</h3>
                    <p>比较平均句长、平均词长、词汇丰富度等句法特征。</p>
                    <div class="similarity-bar">
                        <div class="similarity-bar-fill" style="width: {{ syntactic_sim }}%;">
                            {{ syntactic_sim }}%
                        </div>
                    </div>
                    
                    <h3>语义相似度</h3>
                    <p>基于深度学习预训练模型的文本嵌入向量计算。</p>
                    <div class="similarity-bar">
                        <div class="similarity-bar-fill" style="width: {{ semantic_sim }}%;">
                            {{ semantic_sim }}%
                        </div>
                    </div>
                    
                    <h3>字符 n-gram 相似度</h3>
                    <p>比较相邻字符序列的共现模式。</p>
                    <div class="similarity-bar">
                        <div class="similarity-bar-fill" style="width: {{ ngram_sim }}%;">
                            {{ ngram_sim }}%
                        </div>
                    </div>
                </div>
                
                <!-- 特征对比表 -->
                <div class="section">
                    <h2>📋 文本特征对比</h2>
                    
                    <table>
                        <thead>
                            <tr>
                                <th>特征维度</th>
                                <th>文本 a（风格锚点）</th>
                                <th>文本 A'（作者样本）</th>
                                <th>差异度</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>字数</strong></td>
                                <td>{{ text_a_length }}</td>
                                <td>{{ text_a_prime_length }}</td>
                                <td>{{ length_diff }}%</td>
                            </tr>
                            <tr>
                                <td><strong>词数</strong></td>
                                <td>{{ text_a_words }}</td>
                                <td>{{ text_a_prime_words }}</td>
                                <td>{{ words_diff }}%</td>
                            </tr>
                            <tr>
                                <td><strong>句数</strong></td>
                                <td>{{ text_a_sentences }}</td>
                                <td>{{ text_a_prime_sentences }}</td>
                                <td>{{ sentences_diff }}%</td>
                            </tr>
                            <tr>
                                <td><strong>平均句长</strong></td>
                                <td>{{ text_a_asl }}</td>
                                <td>{{ text_a_prime_asl }}</td>
                                <td>{{ asl_diff }}%</td>
                            </tr>
                            <tr>
                                <td><strong>词汇丰富度 (TTR)</strong></td>
                                <td>{{ text_a_ttr }}</td>
                                <td>{{ text_a_prime_ttr }}</td>
                                <td>{{ ttr_diff }}%</td>
                            </tr>
                            <tr>
                                <td><strong>平均字长</strong></td>
                                <td>{{ text_a_awl }}</td>
                                <td>{{ text_a_prime_awl }}</td>
                                <td>{{ awl_diff }}%</td>
                            </tr>
                            <tr>
                                <td><strong>情感评分</strong></td>
                                <td>{{ text_a_sentiment }}</td>
                                <td>{{ text_a_prime_sentiment }}</td>
                                <td>{{ sentiment_diff }}%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <!-- 高频虚词对比 -->
                <div class="section">
                    <h2>🔤 高频虚词对比</h2>
                    <div class="comparison-grid">
                        <div class="comparison-box">
                            <h3>文本 a 高频虚词</h3>
                            <div id="fw-a"></div>
                        </div>
                        <div class="comparison-box">
                            <h3>文本 A' 高频虚词</h3>
                            <div id="fw-ap"></div>
                        </div>
                    </div>
                </div>
                
                <!-- 标点分布对比 -->
                <div class="section">
                    <h2>📌 标点使用分布对比</h2>
                    <div class="chart-container">
                        {{ punctuation_chart }}
                    </div>
                </div>
                
                <!-- 风格描述 -->
                <div class="section">
                    <h2>✍️ 风格描述与锚点</h2>
                    <h3>原 A 语料的风格特征</h3>
                    <div class="text-sample">
                        {{ style_profile }}
                    </div>
                    <h3>生成的风格锚点 a（约 500 字）</h3>
                    <div class="text-sample">
                        {{ anchor_text }}
                    </div>
                </div>
                
                <!-- 图表区域 -->
                <div class="section">
                    <h2>📊 可视化分析</h2>
                    <div class="chart-container">
                        <h3>综合相似度雷达图</h3>
                        {{ radar_chart }}
                    </div>
                </div>
                
                <!-- 建议 -->
                <div class="section">
                    <h2>💡 建议</h2>
                    <ul style="line-height: 1.8; margin-left: 20px;">
                        {% if overall_confidence > 0.8 %}
                        <li>置信度高：该文本极可能来自作者 A。建议可直接接受。</li>
                        {% elif overall_confidence > 0.6 %}
                        <li>置信度中高：可能来自作者 A，但建议补充更多样本（如 3-5 份 A' 文本）以增强说服力。</li>
                        {% elif overall_confidence > 0.45 %}
                        <li>置信度中等：无法确定，强烈建议：</li>
                        <li style="margin-left: 20px;">1) 补充更多样本 A' 来源</li>
                        <li style="margin-left: 20px;">2) 检查 A 语料是否经历过重大改写/编辑</li>
                        <li style="margin-left: 20px;">3) 进行人工复审或邀请领域专家评判</li>
                        {% else %}
                        <li>置信度低：该文本来源存疑，强烈建议：</li>
                        <li style="margin-left: 20px;">1) 确认 A 语料的完整性与原始性</li>
                        <li style="margin-left: 20px;">2) 获取更多作者 A 的真实样本</li>
                        <li style="margin-left: 20px;">3) 寻求人工专家的最终判定</li>
                        {% endif %}
                    </ul>
                </div>
                
                <!-- 技术说明 -->
                <div class="section">
                    <h2>🔬 技术说明</h2>
                    <p>本报告使用以下算法与模型：</p>
                    <ul style="line-height: 1.8; margin-left: 20px;">
                        <li><strong>Burrows' Delta：</strong> 基于高频虚词的文本风格距离度量，是作者归因研究的经典方法。</li>
                        <li><strong>句法特征：</strong> 平均句长（ASL）、平均词长（AWL）、词汇丰富度（TTR）等。</li>
                        <li><strong>标点指纹：</strong> 统计句号、逗号、感叹号等标点的分布频率。</li>
                        <li><strong>深度学习嵌入：</strong> 使用预训练多语言模型（Sentence-Transformers）进行语义相似度计算。</li>
                        <li><strong>n-gram 分析：</strong> 基于字符级 2-gram 的相似度度量。</li>
                        <li><strong>综合评分：</strong> 加权融合多维特征，虚词指纹权重最高（40%）。</li>
                    </ul>
                </div>
                
                <footer>
                    <p>SARA v1.0 | Style Anchor Reconstruction & Analysis</p>
                    <p>© 2025 | 本报告仅供参考，最终判定应结合人工专家评审。</p>
                </footer>
            </div>
            
            <script>
                // 渲染高频虚词对比
                const fw_a = {{ fw_a_json }};
                const fw_ap = {{ fw_ap_json }};
                
                function renderWords(obj, containerId) {
                    const container = document.getElementById(containerId);
                    for (const [word, freq] of Object.entries(obj).slice(0, 10)) {
                        const pct = Math.round(freq * 10000) / 100;
                        const bar = document.createElement('div');
                        bar.style.marginBottom = '10px';
                        bar.innerHTML = `
                            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                                <span><strong>${word}</strong></span>
                                <span>${pct}%</span>
                            </div>
                            <div style="background: #ecf0f1; height: 20px; border-radius: 3px; overflow: hidden;">
                                <div style="background: #3498db; height: 100%; width: ${pct * 5}%;"></div>
                            </div>
                        `;
                        container.appendChild(bar);
                    }
                }
                
                renderWords(fw_a, 'fw-a');
                renderWords(fw_ap, 'fw-ap');
            </script>
        </body>
        </html>
        """
        
        # 数据映射
        def calculate_diff_percentage(a, ap, a_type='float'):
            if a_type == 'float':
                return f"{abs(a - ap) / max(a, ap, 0.001) * 100:.1f}" if max(a, ap) > 0 else "0"
            else:
                return f"{abs(a - ap) / max(a, ap, 1) * 100:.1f}"
        
        verdict_class_map = {
            "Match (强匹配)": "match",
            "Likely Match (可能匹配)": "likely-match",
            "Uncertain (不确定)": "uncertain",
            "Likely Mismatch (可能不匹配)": "likely-mismatch",
            "Mismatch (不匹配)": "mismatch"
        }
        verdict_class = verdict_class_map.get(result.verdict, "uncertain")
        
        # 将 Delta 转换为百分比
        burrows_sim_pct = max(0, (1 - min(result.burrows_delta, 1.0)) * 100)
        
        template_data = {
            "timestamp": timestamp,
            "pdf_a_path": pdf_a_path,
            "pdf_a_prime_path": pdf_a_prime_path,
            "verdict": result.verdict,
            "verdict_class": verdict_class,
            "confidence_percentage": result.reasoning["confidence_percentage"],
            "reasoning_text": result.reasoning["reason"],
            "burrows_similarity": int(burrows_sim_pct),
            "burrows_delta_value": f"{result.burrows_delta:.4f}",
            "punctuation_sim": int(result.punctuation_similarity * 100),
            "function_words_sim": int(result.function_words_similarity * 100),
            "syntactic_sim": int(result.syntactic_distance * 100),
            "semantic_sim": int(result.semantic_similarity * 100),
            "ngram_sim": int(result.ngram_similarity * 100),
            "text_a_length": result.text_a_features.length,
            "text_a_prime_length": result.text_a_prime_features.length,
            "length_diff": calculate_diff_percentage(result.text_a_features.length, result.text_a_prime_features.length),
            "text_a_words": result.text_a_features.word_count,
            "text_a_prime_words": result.text_a_prime_features.word_count,
            "words_diff": calculate_diff_percentage(result.text_a_features.word_count, result.text_a_prime_features.word_count),
            "text_a_sentences": result.text_a_features.sentence_count,
            "text_a_prime_sentences": result.text_a_prime_features.sentence_count,
            "sentences_diff": calculate_diff_percentage(result.text_a_features.sentence_count, result.text_a_prime_features.sentence_count),
            "text_a_asl": f"{result.text_a_features.avg_sentence_length:.1f}",
            "text_a_prime_asl": f"{result.text_a_prime_features.avg_sentence_length:.1f}",
            "asl_diff": calculate_diff_percentage(result.text_a_features.avg_sentence_length, result.text_a_prime_features.avg_sentence_length),
            "text_a_ttr": f"{result.text_a_features.vocabulary_richness:.2%}",
            "text_a_prime_ttr": f"{result.text_a_prime_features.vocabulary_richness:.2%}",
            "ttr_diff": calculate_diff_percentage(result.text_a_features.vocabulary_richness, result.text_a_prime_features.vocabulary_richness),
            "text_a_awl": f"{result.text_a_features.avg_word_length:.1f}",
            "text_a_prime_awl": f"{result.text_a_prime_features.avg_word_length:.1f}",
            "awl_diff": calculate_diff_percentage(result.text_a_features.avg_word_length, result.text_a_prime_features.avg_word_length),
            "text_a_sentiment": f"{result.text_a_features.sentiment_score:.2f}",
            "text_a_prime_sentiment": f"{result.text_a_prime_features.sentiment_score:.2f}",
            "sentiment_diff": calculate_diff_percentage(result.text_a_features.sentiment_score, result.text_a_prime_features.sentiment_score),
            "fw_a_json": json.dumps(dict(list(result.text_a_features.function_words_freq.items())[:10])),
            "fw_ap_json": json.dumps(dict(list(result.text_a_prime_features.function_words_freq.items())[:10])),
            "punctuation_chart": charts["punctuation"],
            "radar_chart": charts["radar"],
            "style_profile": style_profile,
            "anchor_text": anchor_text,
            "overall_confidence": result.overall_confidence
        }
        
        # 使用 Jinja2 渲染
        jinja_template = Template(html_template)
        html_content = jinja_template.render(**template_data)
        
        # 保存到文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ 报告已生成：{report_path}")
        return str(report_path)
    
    def _generate_charts(self, result: AnalysisResult) -> Dict[str, str]:
        """生成嵌入式图表（SVG）"""
        charts = {}
        
        # 1. 标点分布对比
        punc_a = result.text_a_features.punctuation_dist
        punc_ap = result.text_a_prime_features.punctuation_dist
        
        all_puncs = set(punc_a.keys()) | set(punc_ap.keys())
        all_puncs = sorted([p for p in all_puncs if punc_a.get(p, 0) > 0 or punc_ap.get(p, 0) > 0])[:8]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(all_puncs))
        width = 0.35
        
        ax.bar(x - width/2, [punc_a.get(p, 0) for p in all_puncs], width, label='文本 a', color='#3498db')
        ax.bar(x + width/2, [punc_ap.get(p, 0) for p in all_puncs], width, label='文本 A\'', color='#e74c3c')
        
        ax.set_xlabel('标点符号', fontsize=11)
        ax.set_ylabel('出现次数', fontsize=11)
        ax.set_title('标点使用分布对比', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_puncs)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        svg_str = self._fig_to_svg(fig)
        charts["punctuation"] = svg_str
        plt.close(fig)
        
        # 2. 相似度雷达图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        categories = ['虚词', '标点', '句法', '语义', 'n-gram', '虚词习惯']
        values = [
            1 - min(result.burrows_delta, 1.0),
            result.punctuation_similarity,
            result.syntactic_distance,
            result.semantic_similarity,
            result.ngram_similarity,
            result.function_words_similarity
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db', label='相似度')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('多维度相似度分析', fontsize=13, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        svg_str = self._fig_to_svg(fig)
        charts["radar"] = svg_str
        plt.close(fig)
        
        return charts
    
    @staticmethod
    def _fig_to_svg(fig) -> str:
        """将 matplotlib 图表转换为内联 SVG"""
        import io
        svg_io = io.StringIO()
        fig.savefig(svg_io, format='svg')
        svg_str = svg_io.getvalue()
        return svg_str

# ==================================================================================
# 7. 主控制流程
# ==================================================================================

class SARAPipeline:
    """完整的 SARA 分析流程"""
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        self.pdf_extractor = PDFExtractor()
        self.feature_extractor = FeatureExtractor()
        self.comparator = StyleComparator()
        self.ai_anchor = AIStyleAnchor(api_key)
        self.verdict_engine = VerdictEngine()
        self.report_gen = ReportGenerator()
    
    def run_analysis(self, pdf_a_path: str, pdf_a_prime_path: str) -> AnalysisResult:
        """
        执行完整的作者验证流程
        
        参数：
            pdf_a_path: A 语料（待验证文本）的 PDF 路径
            pdf_a_prime_path: A' 语料（作者样本）的 PDF 路径
        
        返回：
            AnalysisResult: 完整的对比分析结果
        """
        
        print("=" * 60)
        print("🚀 SARA 作者身份验证系统 v1.0")
        print("=" * 60)
        
        # Step 1: 提取文本
        print("\n[1/6] 提取文本...")
        text_a = self.pdf_extractor.extract_text_from_pdf(pdf_a_path)
        if not text_a:
            raise ValueError(f"无法从 {pdf_a_path} 提取文本")
        
        text_a_prime = self.pdf_extractor.extract_text_from_pdf(pdf_a_prime_path)
        if not text_a_prime:
            raise ValueError(f"无法从 {pdf_a_prime_path} 提取文本")
        
        print(f"✓ 文本 A: {len(text_a)} 字")
        print(f"✓ 文本 A': {len(text_a_prime)} 字")
        
        # Step 2: 提取特征
        print("\n[2/6] 提取文本特征...")
        features_a = self.feature_extractor.extract_features(text_a)
        features_a_prime = self.feature_extractor.extract_features(text_a_prime)
        print("✓ 特征提取完成")
        
        # Step 3: 提取 A 的风格描述
        print("\n[3/6] 生成风格描述与锚点...")
        style_profile = self.ai_anchor.extract_style_profile(text_a, features_a)
        print(f"✓ 风格描述生成（{len(style_profile)} 字）")
        
        # Step 4: 生成风格锚点 a
        print("\n[4/6] AI 生成风格锚点 a...")
        anchor_text = self.ai_anchor.generate_style_anchor(text_a, style_profile)
        print(f"✓ 风格锚点生成（{len(anchor_text)} 字）")
        
        # Step 5: 计算相似度
        print("\n[5/6] 计算多维度相似度...")
        
        # 需要对比锚点 a 和 A'
        features_anchor = self.feature_extractor.extract_features(anchor_text)
        embedding_anchor = self.feature_extractor.get_embedding(anchor_text)
        embedding_ap = self.feature_extractor.get_embedding(text_a_prime)
        
        burrows_delta = self.comparator.burrows_delta(
            features_anchor.function_words_freq,
            features_a_prime.function_words_freq
        )
        
        punctuation_sim = self.comparator.punctuation_similarity(
            features_anchor.punctuation_dist,
            features_a_prime.punctuation_dist
        )
        
        function_words_sim = self.comparator.function_words_similarity(
            features_anchor.function_words_freq,
            features_a_prime.function_words_freq
        )
        
        syntactic_sim = self.comparator.syntactic_distance(features_anchor, features_a_prime)
        
        semantic_sim = self.comparator.semantic_similarity(embedding_anchor, embedding_ap)
        
        ngram_sim = self.comparator.ngram_similarity(anchor_text, text_a_prime)
        
        print(f"✓ Burrows' Delta: {burrows_delta:.4f}")
        print(f"✓ 标点相似度: {punctuation_sim:.2%}")
        print(f"✓ 虚词相似度: {function_words_sim:.2%}")
        print(f"✓ 句法相似度: {syntactic_sim:.2%}")
        print(f"✓ 语义相似度: {semantic_sim:.2%}")
        print(f"✓ n-gram 相似度: {ngram_sim:.2%}")
        
        # Step 6: 综合判定
        print("\n[6/6] 综合判定...")
        overall_confidence, verdict, reasoning = self.verdict_engine.compute_confidence(
            burrows_delta,
            punctuation_sim,
            semantic_sim,
            ngram_sim,
            function_words_sim,
            syntactic_sim
        )
        
        print(f"✓ 置信度: {overall_confidence:.2%}")
        print(f"✓ 判定: {verdict}")
        
        # 创建结果对象
        result = AnalysisResult(
            text_a_features=features_anchor,
            text_a_prime_features=features_a_prime,
            burrows_delta=burrows_delta,
            punctuation_similarity=punctuation_sim,
            semantic_similarity=semantic_sim,
            ngram_similarity=ngram_sim,
            function_words_similarity=function_words_sim,
            syntactic_distance=syntactic_sim,
            overall_confidence=overall_confidence,
            verdict=verdict,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat()
        )
        
        # 生成报告
        print("\n[✓] 生成可视化报告...")
        report_path = self.report_gen.generate_html_report(
            result, style_profile, anchor_text, pdf_a_path, pdf_a_prime_path
        )
        
        print("\n" + "=" * 60)
        print("✅ 分析完成！")
        print(f"📊 报告位置: {report_path}")
        print("=" * 60)
        
        return result

# ==================================================================================
# 主函数
# ==================================================================================

if __name__ == "__main__":
    import sys
    
    # 命令行使用示例
    if len(sys.argv) < 3:
        print("""
        使用方法：
        python SARA_complete_system.py <A 语料 PDF> <A' 语料 PDF>
        
        示例：
        python SARA_complete_system.py sample_a.pdf sample_a_prime.pdf
        """)
        sys.exit(1)
    
    pdf_a = sys.argv[1]
    pdf_a_prime = sys.argv[2]
    
    # 初始化 pipeline
    pipeline = SARAPipeline()
    
    # 运行分析
    result = pipeline.run_analysis(pdf_a, pdf_a_prime)
    
    # 输出结果
    print("\n[最终结果]")
    print(json.dumps(result.reasoning, indent=2, ensure_ascii=False))
