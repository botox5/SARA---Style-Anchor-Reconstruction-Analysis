# -*- coding: utf-8 -*-
import os
import re
import json
import math
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime

import numpy as np
import pdfplumber
import jieba
from jieba import posseg as pseg

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from openai import OpenAI
from jinja2 import Template

# -----------------------------
# Config
# -----------------------------
DEFAULT_FUNCTION_WORDS_ZH = [
    "的","了","和","是","在","这","那","我","你","他","她","它",
    "但是","所以","因为","而","或","以及","并且","而且","然而","却","不过",
    "其实","可能","或许","似乎","应该","必须","需要","可以","能够","无法","不能",
    "也","都","只","就","还","再","又","很","太","非常","不是","没有","没","无",
    "对","把","被","让","给","从","到","向","于","之"
]

DEFAULT_PUNCS = ["。","，","！","？","；","、","：","（","）","“","”","——","…","-"]

@dataclass
class PreparePackage:
    version: str
    created_at: str
    source_pdf: str
    cleaned_text_sha256: str
    cleaned_text_preview: str
    style_profile: str
    anchor_text_a: str
    a_prime_prompt: str
    config: Dict

@dataclass
class TextFeatures:
    length_chars: int
    word_count: int
    sentence_count: int
    avg_sentence_len_words: float
    ttr: float
    avg_word_len_chars: float
    function_word_freq: Dict[str, float]      # relative freq
    punctuation_count: Dict[str, int]
    char_2gram_jaccard: float                 # filled later if needed
    embedding: Optional[List[float]]          # stored optionally

@dataclass
class CompareMetrics:
    burrows_delta: float
    function_words_similarity: float
    punctuation_similarity: float
    syntactic_similarity: float
    semantic_similarity: float
    char_2gram_similarity: float
    overall_confidence: float
    verdict: str

# -----------------------------
# PDF extraction & cleaning
# -----------------------------
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def remove_references_section(text: str) -> str:
    # 简易：出现“参考文献/References/致谢”等后截断
    m = re.search(r"(参考文献|References|REFERENCES|致谢|Acknowledg(e)?ments?)\b[\s\S]*$", text)
    if m:
        return text[:m.start()].strip()
    return text

def clean_text(text: str, drop_urls=True, drop_page_numbers=True) -> str:
    if drop_urls:
        text = re.sub(r"http[s]?://\S+", "", text)
    if drop_page_numbers:
        text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.M)
        text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", text, flags=re.M)
    return _normalize_whitespace(text)

def obj_in_bbox(obj_bbox, table_bbox) -> bool:
    # bbox: (x0, top, x1, bottom) in pdfplumber
    ox0, otop, ox1, obot = obj_bbox
    tx0, ttop, tx1, tbot = table_bbox
    # intersection check
    return not (ox1 < tx0 or ox0 > tx1 or obot < ttop or otop > tbot)

class PDFExtractor:
    def __init__(self, remove_tables: bool = True):
        self.remove_tables = remove_tables

    def extract_text(self, pdf_path: str) -> str:
        chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                p = page
                if self.remove_tables:
                    # 基于表格 bbox 过滤表格区域对象的常见做法
                    try:
                        tables = p.find_tables()
                        bboxes = [t.bbox for t in tables] if tables else []
                        if bboxes:
                            def _filter(obj):
                                if "x0" in obj and "top" in obj and "x1" in obj and "bottom" in obj:
                                    ob = (obj["x0"], obj["top"], obj["x1"], obj["bottom"])
                                    return not any(obj_in_bbox(ob, tb) for tb in bboxes)
                                return True
                            p = p.filter(_filter)
                    except Exception:
                        pass

                txt = p.extract_text() or ""
                if txt.strip():
                    chunks.append(txt)

        return "\n".join(chunks)

# -----------------------------
# Feature extraction
# -----------------------------
class FeatureExtractor:
    def __init__(self, function_words: Optional[List[str]] = None, embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.function_words = function_words or DEFAULT_FUNCTION_WORDS_ZH
        self.embedding_model = None
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception:
            self.embedding_model = None

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"[。！？\n]+", text)
        return [p.strip() for p in parts if p.strip()]

    def _punct_counts(self, text: str) -> Dict[str, int]:
        return {p: text.count(p) for p in DEFAULT_PUNCS}

    def _function_word_freq(self, words: List[str]) -> Dict[str, float]:
        total = max(len(words), 1)
        cnt = Counter(words)
        freq = {w: cnt.get(w, 0) / total for w in self.function_words}
        # keep top 50 by freq (or all if shorter)
        top = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:50])
        return top

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not self.embedding_model:
            return None
        t = text[:2000]  # avoid super long
        vec = self.embedding_model.encode(t, convert_to_numpy=True)
        return vec

    def extract(self, text: str, store_embedding: bool = False) -> TextFeatures:
        text = text.strip()
        sents = self._split_sentences(text)
        words = [w for w in jieba.lcut(text) if w.strip()]
        wc = len(words)
        sc = len(sents) if sents else 1
        avg_sent = wc / sc if sc else 0.0
        ttr = len(set(words)) / max(wc, 1)
        avg_word_len = len(text) / max(wc, 1)

        fw = self._function_word_freq(words)
        pc = self._punct_counts(text)
        emb = self.embed(text)
        emb_list = emb.tolist() if (store_embedding and emb is not None) else None

        return TextFeatures(
            length_chars=len(text),
            word_count=wc,
            sentence_count=sc,
            avg_sentence_len_words=float(avg_sent),
            ttr=float(ttr),
            avg_word_len_chars=float(avg_word_len),
            function_word_freq=fw,
            punctuation_count=pc,
            char_2gram_jaccard=0.0,
            embedding=emb_list
        )

# -----------------------------
# Similarity / distance
# -----------------------------
def cosine_sim_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a.keys()) | set(b.keys()))
    va = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.5
    return float(np.dot(va, vb) / (na * nb))

def punctuation_cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    keys = sorted(set(a.keys()) | set(b.keys()))
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.5
    return float(np.dot(va, vb) / (na * nb))

def char_ngrams(text: str, n: int = 2) -> Counter:
    text = re.sub(r"\s+", "", text)
    return Counter(text[i:i+n] for i in range(max(len(text) - n + 1, 0)))

def jaccard_multiset(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.5
    inter = sum((a & b).values())
    uni = sum((a | b).values())
    return float(inter / uni) if uni else 0.5

def syntactic_similarity(fa: TextFeatures, fb: TextFeatures) -> float:
    def rel_diff(x, y, eps=1e-9):
        return abs(x - y) / max(abs(x), abs(y), eps)
    d1 = rel_diff(fa.avg_sentence_len_words, fb.avg_sentence_len_words, 1.0)
    d2 = rel_diff(fa.avg_word_len_chars, fb.avg_word_len_chars, 1.0)
    d3 = abs(fa.ttr - fb.ttr)
    dist = (d1 + d2 + d3) / 3.0
    return float(max(0.0, 1.0 - min(dist, 1.0)))

def semantic_similarity(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> float:
    if vec_a is None or vec_b is None:
        return 0.5
    s = float(cosine_similarity([vec_a], [vec_b])[0][0])
    return (s + 1) / 2  # to [0,1]

def burrows_delta_chunked(text_a: str, text_b: str, top_k: int = 50, chunks: int = 8) -> float:
    """
    更稳健的 Delta：把每段文本切成 chunks 份，按每个 chunk 的高频词频率估计均值/方差，再做 z-score 差异。
    """
    def chunkify(t: str, k: int) -> List[str]:
        t = re.sub(r"\s+", "", t)
        if len(t) < k:
            return [t]
        size = max(1, len(t) // k)
        return [t[i:i+size] for i in range(0, len(t), size)][:k]

    def fw_freq(t: str, vocab: List[str]) -> Dict[str, float]:
        words = [w for w in jieba.lcut(t) if w.strip()]
        total = max(len(words), 1)
        c = Counter(words)
        return {w: c.get(w, 0) / total for w in vocab}

    # build vocab by overall most frequent among both
    words_all = [w for w in jieba.lcut(text_a + "\n" + text_b) if w.strip()]
    common = [w for (w, _) in Counter(words_all).most_common(500)]
    # prefer function-like: short tokens
    common = [w for w in common if len(w) <= 2]
    vocab = common[:top_k] if common else DEFAULT_FUNCTION_WORDS_ZH[:top_k]

    ca = chunkify(text_a, chunks)
    cb = chunkify(text_b, chunks)
    fa = [fw_freq(x, vocab) for x in ca]
    fb = [fw_freq(x, vocab) for x in cb]

    # pooled mean/std per word
    delta_terms = []
    for w in vocab:
        vals = [d[w] for d in fa] + [d[w] for d in fb]
        mu = float(np.mean(vals))
        sd = float(np.std(vals)) + 1e-9
        za = (float(np.mean([d[w] for d in fa])) - mu) / sd
        zb = (float(np.mean([d[w] for d in fb])) - mu) / sd
        delta_terms.append(abs(za - zb))

    return float(np.mean(delta_terms)) if delta_terms else 0.0

# -----------------------------
# LLM (GPT-4) steps: style profile, anchor, A' prompt
# -----------------------------
class LLMService:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=key)
        self.model = model

    def style_profile(self, text_a: str) -> str:
        prompt = f"""
请对以下中文文本的写作风格做“可操作”的画像，输出 180-260 字，尽量量化与可复用。
需要包含：
1) 句法节奏（长短句、并列/从句倾向）
2) 常用连接词/虚词口癖（举例 8-12 个）
3) 标点与段落习惯
4) 语气（克制/锋利/抒情/学术等）
5) 论证结构（如：提出问题→拆解→结论；或：先例后论）

文本（截取前 1800 字）：
{text_a[:1800]}
"""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=550
        )
        return r.choices[0].message.content.strip()

    def generate_anchor_a(self, text_a: str, style_profile: str) -> str:
        prompt = f"""
你是专业的“风格保持型摘要/改写”专家。
任务：把原文压缩改写为约 500 字的“风格锚点 a”。

要求：
- 450-550 字。
- 保留原文核心主张/逻辑链，不要堆砌细节。
- 严格继承下述风格画像（连接词、句式、标点节奏、语气、段落开合）。
- 禁止出现 AI 套话（如“综上所述”），除非原文风格本来就会用。

风格画像：
{style_profile}

原文（截取前 3500 字用于生成）：
{text_a[:3500]}

只输出“a”文本本体，不要任何解释。
"""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=900
        )
        return r.choices[0].message.content.strip()

    def generate_a_prime_prompt(self, anchor_a: str) -> str:
        """
        Step [5]：根据 a 的主题、内容结构生成作者写作提示词（让作者写出 A'）
        """
        prompt = f"""
你将得到一段“风格锚点 a”。请生成一份给作者的写作任务书，用于产出 A'（约 500 字）。
要求：
1) A' 必须围绕与 a “同主题/同问题域”，但不得复述 a 的具体措辞或例子（避免抄写）。
2) 必须复刻 a 的结构动作：段落数、每段功能（引题/论证/反思收束等）、论证推进方式。
3) 明确字数范围（450-550 字）、时间限制建议（20-30 分钟）、禁止事项（查资料/复制句子/模板化套话）。
4) 输出格式：标题 + 任务要求要点（bullet）+ 可选写作提纲（3-4 条）。

风格锚点 a：
{anchor_a}
"""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=700
        )
        return r.choices[0].message.content.strip()

# -----------------------------
# Verdict & report
# -----------------------------
def fuse_confidence(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    s = 0.0
    for k, w in weights.items():
        s += metrics.get(k, 0.5) * w
    return float(max(0.0, min(1.0, s)))

def verdict_from_confidence(conf: float) -> str:
    if conf >= 0.80:
        return "Match"
    if conf >= 0.60:
        return "Likely Match"
    if conf >= 0.45:
        return "Uncertain"
    if conf >= 0.30:
        return "Likely Mismatch"
    return "Mismatch"

class ReportGenerator:
    def __init__(self):
        pass

    def render_html(self, package: PreparePackage, metrics: CompareMetrics, fa: TextFeatures, fb: TextFeatures) -> str:
        tpl = Template("""
<!doctype html><html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SARA v2 报告</title>
<style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;background:#f6f7fb;margin:0;padding:24px;color:#111}
.card{background:#fff;border-radius:12px;padding:18px;margin:12px 0;box-shadow:0 4px 16px rgba(0,0,0,.06)}
h1{margin:0 0 8px 0}
small{color:#555}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;background:#eef}
.kpi{font-size:28px;font-weight:700}
table{width:100%;border-collapse:collapse}
td,th{padding:8px;border-bottom:1px solid #eee;text-align:left}
pre{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;padding:10px;border-radius:10px}
.bar{height:10px;background:#eee;border-radius:999px;overflow:hidden}
.fill{height:10px;background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71)}
</style>
</head>
<body>
  <div class="card">
    <h1>SARA v2 作者验证报告</h1>
    <small>生成时间：{{ now }} | 版本：{{ package.version }}</small>
    <div style="margin-top:10px">
      <span class="badge">结论：{{ metrics.verdict }}</span>
      <span class="badge">置信度：{{ (metrics.overall_confidence*100)|round(1) }}%</span>
    </div>
    <div style="margin-top:12px" class="bar"><div class="fill" style="width:{{ (metrics.overall_confidence*100)|round(1) }}%"></div></div>
  </div>

  <div class="card">
    <h2>输入与阶段化产物</h2>
    <table>
      <tr><th>项目</th><th>内容</th></tr>
      <tr><td>A PDF</td><td>{{ package.source_pdf }}</td></tr>
      <tr><td>A 清理后摘要 hash</td><td><code>{{ package.cleaned_text_sha256 }}</code></td></tr>
      <tr><td>a（风格锚点）长度</td><td>{{ package.anchor_text_a|length }} chars</td></tr>
      <tr><td>A' 写作提示词</td><td>见下方</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>多维对比指标</h2>
    <table>
      <tr><th>维度</th><th>值</th></tr>
      <tr><td>Burrows Delta（越小越像）</td><td>{{ metrics.burrows_delta|round(4) }}</td></tr>
      <tr><td>虚词相似度</td><td>{{ (metrics.function_words_similarity*100)|round(1) }}%</td></tr>
      <tr><td>标点相似度</td><td>{{ (metrics.punctuation_similarity*100)|round(1) }}%</td></tr>
      <tr><td>句法相似度</td><td>{{ (metrics.syntactic_similarity*100)|round(1) }}%</td></tr>
      <tr><td>语义相似度</td><td>{{ (metrics.semantic_similarity*100)|round(1) }}%</td></tr>
      <tr><td>字符 2-gram 相似度</td><td>{{ (metrics.char_2gram_similarity*100)|round(1) }}%</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>a 与 A' 基础特征</h2>
    <div class="grid">
      <div>
        <h3>a（锚点）</h3>
        <table>
          <tr><td>词数</td><td>{{ fa.word_count }}</td></tr>
          <tr><td>句数</td><td>{{ fa.sentence_count }}</td></tr>
          <tr><td>平均句长</td><td>{{ fa.avg_sentence_len_words|round(2) }}</td></tr>
          <tr><td>TTR</td><td>{{ (fa.ttr*100)|round(2) }}%</td></tr>
        </table>
      </div>
      <div>
        <h3>A'（作者样本）</h3>
        <table>
          <tr><td>词数</td><td>{{ fb.word_count }}</td></tr>
          <tr><td>句数</td><td>{{ fb.sentence_count }}</td></tr>
          <tr><td>平均句长</td><td>{{ fb.avg_sentence_len_words|round(2) }}</td></tr>
          <tr><td>TTR</td><td>{{ (fb.ttr*100)|round(2) }}%</td></tr>
        </table>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>风格描述（A）</h2>
    <pre>{{ package.style_profile }}</pre>
  </div>

  <div class="card">
    <h2>a（风格锚点）</h2>
    <pre>{{ package.anchor_text_a }}</pre>
  </div>

  <div class="card">
    <h2>A' 写作提示词（给作者）</h2>
    <pre>{{ package.a_prime_prompt }}</pre>
  </div>

  <div class="card">
    <h2>A 清理后预览</h2>
    <pre>{{ package.cleaned_text_preview }}</pre>
  </div>
</body>
</html>
""")
        return tpl.render(
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            package=package,
            metrics=metrics,
            fa=fa,
            fb=fb
        )

# -----------------------------
# Pipeline functions
# -----------------------------
def prepare_from_a_pdf(
    a_pdf: str,
    out_dir: str,
    model: str = "gpt-4o",
    remove_tables: bool = True,
    remove_references: bool = True
) -> PreparePackage:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    extractor = PDFExtractor(remove_tables=remove_tables)
    raw = extractor.extract_text(a_pdf)
    cleaned = clean_text(raw)
    if remove_references:
        cleaned = remove_references_section(cleaned)

    llm = LLMService(model=model)
    style = llm.style_profile(cleaned)
    anchor = llm.generate_anchor_a(cleaned, style)
    a_prime_prompt = llm.generate_a_prime_prompt(anchor)

    pkg = PreparePackage(
        version="sara-v2",
        created_at=datetime.now().isoformat(),
        source_pdf=str(a_pdf),
        cleaned_text_sha256=_sha256_text(cleaned),
        cleaned_text_preview=cleaned[:1200],
        style_profile=style,
        anchor_text_a=anchor,
        a_prime_prompt=a_prime_prompt,
        config={
            "model": model,
            "remove_tables": remove_tables,
            "remove_references": remove_references,
        }
    )

    (out / "package.json").write_text(json.dumps(asdict(pkg), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "a.txt").write_text(anchor, encoding="utf-8")
    (out / "style_profile.txt").write_text(style, encoding="utf-8")
    (out / "a_prime_prompt.txt").write_text(a_prime_prompt, encoding="utf-8")
    (out / "A_cleaned_preview.txt").write_text(pkg.cleaned_text_preview, encoding="utf-8")

    return pkg

def verify_with_a_prime_pdf(
    package_json: str,
    a_prime_pdf: str,
    out_dir: str,
    remove_tables: bool = True,
    remove_references: bool = True
) -> Tuple[CompareMetrics, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pkg_dict = json.loads(Path(package_json).read_text(encoding="utf-8"))
    pkg = PreparePackage(**pkg_dict)

    extractor = PDFExtractor(remove_tables=remove_tables)
    raw_ap = extractor.extract_text(a_prime_pdf)
    cleaned_ap = clean_text(raw_ap)
    if remove_references:
        cleaned_ap = remove_references_section(cleaned_ap)

    feat = FeatureExtractor()
    fa = feat.extract(pkg.anchor_text_a)
    fb = feat.extract(cleaned_ap)

    # components
    delta = burrows_delta_chunked(pkg.anchor_text_a, cleaned_ap, top_k=50, chunks=8)  # smaller is closer
    # map delta to similarity roughly: sim = 1/(1+delta)
    delta_sim = 1.0 / (1.0 + delta)

    fw_sim = cosine_sim_dict(fa.function_word_freq, fb.function_word_freq)
    p_sim = punctuation_cosine(fa.punctuation_count, fb.punctuation_count)
    syn_sim = syntactic_similarity(fa, fb)

    ea = feat.embed(pkg.anchor_text_a)
    eb = feat.embed(cleaned_ap)
    sem_sim = semantic_similarity(ea, eb)

    ng_a = char_ngrams(pkg.anchor_text_a, 2)
    ng_b = char_ngrams(cleaned_ap, 2)
    ng_sim = jaccard_multiset(ng_a, ng_b)

    # fuse (weights aligned to your “6 维度”，并把 delta_sim 作为“虚词指纹核心”)
    weights = {
        "delta_sim": 0.40,
        "fw_sim": 0.20,
        "p_sim": 0.15,
        "syn_sim": 0.15,
        "sem_sim": 0.05,
        "ng_sim": 0.05
    }
    conf = fuse_confidence(
        {"delta_sim": delta_sim, "fw_sim": fw_sim, "p_sim": p_sim, "syn_sim": syn_sim, "sem_sim": sem_sim, "ng_sim": ng_sim},
        weights
    )
    verdict = verdict_from_confidence(conf)

    metrics = CompareMetrics(
        burrows_delta=float(delta),
        function_words_similarity=float(fw_sim),
        punctuation_similarity=float(p_sim),
        syntactic_similarity=float(syn_sim),
        semantic_similarity=float(sem_sim),
        char_2gram_similarity=float(ng_sim),
        overall_confidence=float(conf),
        verdict=verdict
    )

    # report
    html = ReportGenerator().render_html(pkg, metrics, fa, fb)
    report_path = out / "SARA_v2_Report.html"
    report_path.write_text(html, encoding="utf-8")

    # result json
    (out / "result.json").write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics, str(report_path)
