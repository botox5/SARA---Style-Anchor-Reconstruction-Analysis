# ==================================================================================
# SARA 系统 - 快速测试脚本
# 用于本地测试和演示，无需真实 PDF 文件
# ==================================================================================

import os
import sys
from pathlib import Path
from SARA_complete_system import (
    SARAPipeline, FeatureExtractor, StyleComparator, VerdictEngine, ReportGenerator
)

# ==================================================================================
# 1. 内存测试（不依赖 PDF 文件）
# ==================================================================================

def test_with_sample_texts():
    """
    使用内存中的示例文本进行测试
    这样可以在没有 PDF 的情况下测试整个流程
    """
    
    print("=" * 70)
    print("🧪 SARA 系统 - 快速测试模式")
    print("=" * 70)
    
    # 示例文本 A（待验证）
    text_a = """
    在当今数字化时代，人工智能技术正以前所未有的速度改变着我们的生活方式。
    但是，我们不能忽视一个重要的问题：技术发展与人类伦理的平衡。
    
    首先，人工智能在医疗领域取得了显著成效。通过深度学习算法，医生可以更准确地
    诊断疾病。然而，这也引发了一些关于隐私保护的疑虑。患者的健康数据被广泛使用，
    其实这需要更严格的法律监管。
    
    其次，教育领域也面临着重大变革。人工智能个性化学习系统能够根据学生的特点
    提供定制化教学。但是，我们必须思考：机器能否真正替代教师的角色？答案显然是否定的。
    教育的核心是人文关怀，这正是机器无法提供的。
    
    最后，我想强调的是，人工智能发展应该始终以人为本。无论技术如何进步，
    我们都不应该忽视人类的价值和尊严。所以，建立一套完整的伦理框架是当务之急。
    """
    
    # 示例文本 A'（作者的新样本）
    text_a_prime = """
    云计算技术在互联网产业中扮演着越来越重要的角色。但是，云计算的安全问题
    仍然是制约其广泛应用的关键因素。其实，数据泄露事件频繁发生，这说明了
    现有的安全机制还远远不够完善。
    
    首先，云计算提供商必须投入大量资源用于安全防护。然而，在利润驱动下，
    很多企业对此并不积极。这是一个矛盾的现象，值得我们深入思考。
    
    其次，用户数据隐私保护应该成为第一优先级。我们看到欧洲的 GDPR 法规
    在这方面做出了重要尝试。但是，全球化背景下，单一地区的法规显然不够。
    所以，国际合作变得尤为重要。
    
    最后，我要指出的是，技术安全与商业利益之间的平衡点需要通过立法来确定。
    无论企业如何抱怨，保护用户权益都应该是首要责任。因此，建立全球统一的
    数据保护标准势在必行。
    """
    
    print("\n✓ 加载示例文本")
    print(f"  - 文本 A：{len(text_a)} 字")
    print(f"  - 文本 A'：{len(text_a_prime)} 字")
    
    # 实例化组件
    print("\n✓ 初始化 SARA 组件")
    feature_extractor = FeatureExtractor()
    comparator = StyleComparator()
    verdict_engine = VerdictEngine()
    
    # 提取特征
    print("\n✓ 特征提取...")
    features_a = feature_extractor.extract_features(text_a)
    features_ap = feature_extractor.extract_features(text_a_prime)
    
    print(f"\n  文本 A 特征:")
    print(f"    - 字数: {features_a.word_count}")
    print(f"    - 句数: {features_a.sentence_count}")
    print(f"    - 平均句长: {features_a.avg_sentence_length:.1f}")
    print(f"    - 词汇丰富度: {features_a.vocabulary_richness:.2%}")
    
    print(f"\n  文本 A' 特征:")
    print(f"    - 字数: {features_ap.word_count}")
    print(f"    - 句数: {features_ap.sentence_count}")
    print(f"    - 平均句长: {features_ap.avg_sentence_length:.1f}")
    print(f"    - 词汇丰富度: {features_ap.vocabulary_richness:.2%}")
    
    # 计算相似度
    print("\n✓ 计算相似度...")
    
    burrows_delta = comparator.burrows_delta(
        features_a.function_words_freq,
        features_ap.function_words_freq
    )
    
    punctuation_sim = comparator.punctuation_similarity(
        features_a.punctuation_dist,
        features_ap.punctuation_dist
    )
    
    function_words_sim = comparator.function_words_similarity(
        features_a.function_words_freq,
        features_ap.function_words_freq
    )
    
    syntactic_sim = comparator.syntactic_distance(features_a, features_ap)
    
    embedding_a = feature_extractor.get_embedding(text_a)
    embedding_ap = feature_extractor.get_embedding(text_a_prime)
    semantic_sim = comparator.semantic_similarity(embedding_a, embedding_ap)
    
    ngram_sim = comparator.ngram_similarity(text_a, text_a_prime)
    
    print(f"\n  相似度指标:")
    print(f"    - Burrows' Delta: {burrows_delta:.4f}")
    print(f"    - 标点相似度: {punctuation_sim:.2%}")
    print(f"    - 虚词相似度: {function_words_sim:.2%}")
    print(f"    - 句法相似度: {syntactic_sim:.2%}")
    print(f"    - 语义相似度: {semantic_sim:.2%}")
    print(f"    - n-gram 相似度: {ngram_sim:.2%}")
    
    # 综合判定
    print("\n✓ 综合判定...")
    overall_confidence, verdict, reasoning = verdict_engine.compute_confidence(
        burrows_delta,
        punctuation_sim,
        semantic_sim,
        ngram_sim,
        function_words_sim,
        syntactic_sim
    )
    
    print(f"\n  【最终结果】")
    print(f"    - 置信度: {overall_confidence:.2%}")
    print(f"    - 判定: {verdict}")
    print(f"    - 理由: {reasoning['reason']}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)
    
    return overall_confidence, verdict

# ==================================================================================
# 2. 高级特征分析
# ==================================================================================

def test_feature_details():
    """
    详细打印文本的全部提取特征
    """
    
    text = """
    人工智能的发展日新月异。但是，我们不能忽视伦理问题。其实，技术进步
    总是伴随着挑战。首先，安全是首要考虑。然而，利益驱动往往压过安全。
    所以，立法变得迫在眉睫。最后，我想强调的是，道德底线不能突破。
    """
    
    print("\n" + "=" * 70)
    print("🔬 详细特征分析")
    print("=" * 70)
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(text)
    
    print(f"\n【基础统计】")
    print(f"  字数: {features.length}")
    print(f"  词数: {features.word_count}")
    print(f"  句数: {features.sentence_count}")
    print(f"  平均句长: {features.avg_sentence_length:.2f} 字/句")
    print(f"  平均词长: {features.avg_word_length:.2f} 字/词")
    
    print(f"\n【虚词分布】(前 15 个)")
    for i, (word, freq) in enumerate(list(features.function_words_freq.items())[:15], 1):
        print(f"  {i:2d}. '{word}': {freq:.4f} ({freq*10000:.0f}/万字)")
    
    print(f"\n【标点分布】")
    for punct, count in sorted(features.punctuation_dist.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  '{punct}': {count} 次")
    
    print(f"\n【语义特征】")
    print(f"  情感评分: {features.sentiment_score:.2f} (0=负面, 1=正面)")
    print(f"  词汇丰富度 (TTR): {features.vocabulary_richness:.2%}")
    
    if features.named_entities:
        print(f"\n【命名实体】")
        for entity in features.named_entities[:10]:
            print(f"  - {entity}")
    
    print("\n" + "=" * 70)

# ==================================================================================
# 3. 虚词指纹演示
# ==================================================================================

def demonstrate_function_words():
    """
    演示虚词如何成为作者的"指纹"
    """
    
    print("\n" + "=" * 70)
    print("🔐 虚词指纹演示（Burrows' Delta 原理）")
    print("=" * 70)
    
    # 模拟两个不同作者的虚词使用习惯
    text_author_a = "但是这个问题很严重，其实我们都知道。然而很多人不在乎。所以说，"
    text_author_b = "虽然这个问题很复杂，不过我们应该重视。可是大多数人都忽略了。因此，"
    
    extractor = FeatureExtractor()
    
    features_a = extractor.extract_features(text_author_a)
    features_b = extractor.extract_features(text_author_b)
    
    comparator = StyleComparator()
    delta = comparator.burrows_delta(
        features_a.function_words_freq,
        features_b.function_words_freq
    )
    
    print(f"\n【作者 A 虚词使用】")
    print(f"  {text_author_a}")
    
    print(f"\n【作者 B 虚词使用】")
    print(f"  {text_author_b}")
    
    print(f"\n【Burrows' Delta 计算】")
    print(f"  差异指数: {delta:.4f}")
    if delta > 0.5:
        print(f"  结论: 两个作者风格差异显著 (Delta > 0.5)")
    else:
        print(f"  结论: 两个作者风格相近 (Delta < 0.5)")
    
    print("\n说明：Burrows' Delta 基于最常见的虚词（"但是"、"然而"、"所以"等）的")
    print("      频率分布。这些词汇因为潜意识使用，极难伪造，是作者验证的金标准。")
    
    print("\n" + "=" * 70)

# ==================================================================================
# 4. 生成示例 PDF 用于测试
# ==================================================================================

def create_sample_pdf_files():
    """
    生成可用于测试的 PDF 样本文件
    需要 reportlab 库
    """
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
    except ImportError:
        print("⚠️  未安装 reportlab，跳过 PDF 生成。")
        print("   可运行: pip install reportlab")
        return
    
    print("\n" + "=" * 70)
    print("📄 生成示例 PDF 文件")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = Path("./sample_pdfs")
    output_dir.mkdir(exist_ok=True)
    
    text_a = """
    人工智能的未来发展方向。但是，我们必须重视其中的伦理问题。其实，
    每一项技术进步都伴随着相应的风险。首先，数据安全是基础。然而，
    很多企业为了利益而忽视这一点。所以，强有力的法律规制显得尤为重要。
    最后，我想强调的是，技术应该为人类服务，而非相反。
    """
    
    text_a_prime = """
    云计算在现代社会的应用。但是，云计算也带来了新的挑战。其实，
    这些挑战主要来自于安全方面。首先，用户隐私保护至关重要。然而，
    有些服务商并未给予足够重视。所以，国际合作与标准化变得必不可少。
    最后，我认为应该在创新与保护之间找到平衡点。
    """
    
    # 创建 PDF A
    pdf_a_path = output_dir / "sample_text_a.pdf"
    c = canvas.Canvas(str(pdf_a_path), pagesize=letter)
    c.setFont("SimHei", 12)
    y = 750
    for line in text_a.split('\n'):
        if line.strip():
            c.drawString(50, y, line)
            y -= 20
    c.save()
    print(f"✓ 已生成: {pdf_a_path}")
    
    # 创建 PDF A'
    pdf_a_prime_path = output_dir / "sample_text_a_prime.pdf"
    c = canvas.Canvas(str(pdf_a_prime_path), pagesize=letter)
    c.setFont("SimHei", 12)
    y = 750
    for line in text_a_prime.split('\n'):
        if line.strip():
            c.drawString(50, y, line)
            y -= 20
    c.save()
    print(f"✓ 已生成: {pdf_a_prime_path}")
    
    print(f"\n现在可以运行:")
    print(f"  python SARA_complete_system.py sample_pdfs/sample_text_a.pdf sample_pdfs/sample_text_a_prime.pdf")
    
    print("\n" + "=" * 70)

# ==================================================================================
# 5. 压力测试
# ==================================================================================

def stress_test():
    """
    测试系统在大文本上的表现
    """
    
    print("\n" + "=" * 70)
    print("⚡ 性能压力测试")
    print("=" * 70)
    
    import time
    
    # 创建 10000 字的文本
    base_text = """
    人工智能技术正在全面改造各个行业。但是，技术进步往往伴随着挑战。
    其实，我们需要建立完整的监管框架。首先，安全是首要考虑。然而，
    利益驱动往往压过安全。所以，立法变得迫在眉睫。最后，我想强调的是，
    道德底线不能突破。
    """
    
    large_text = base_text * 100  # 重复 100 次
    
    print(f"\n测试文本大小: {len(large_text)} 字")
    
    extractor = FeatureExtractor()
    
    start = time.time()
    features = extractor.extract_features(large_text)
    extract_time = time.time() - start
    
    print(f"特征提取耗时: {extract_time:.2f} 秒")
    print(f"处理速度: {len(large_text) / extract_time:.0f} 字/秒")
    
    if extract_time < 5:
        print("✅ 性能优秀")
    elif extract_time < 10:
        print("⚠️  性能一般")
    else:
        print("❌ 性能需要优化")
    
    print("\n" + "=" * 70)

# ==================================================================================
# 主函数
# ==================================================================================

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        test_type = "all"
    
    if test_type == "all" or test_type == "text":
        test_with_sample_texts()
    
    if test_type == "all" or test_type == "features":
        test_feature_details()
    
    if test_type == "all" or test_type == "function_words":
        demonstrate_function_words()
    
    if test_type == "all" or test_type == "stress":
        stress_test()
    
    if test_type == "pdf":
        create_sample_pdf_files()
    
    print(f"""
    
╔════════════════════════════════════════════════════════════════════╗
║              SARA 系统测试脚本 - 使用说明                          ║
╚════════════════════════════════════════════════════════════════════╝

可用命令：

  python test_sara.py all             # 运行所有测试
  python test_sara.py text            # 文本相似度测试
  python test_sara.py features        # 特征提取详情
  python test_sara.py function_words  # 虚词指纹演示
  python test_sara.py stress          # 性能压力测试
  python test_sara.py pdf             # 生成示例 PDF 文件

正式使用：

  python SARA_complete_system.py <A 语料 PDF> <A' 语料 PDF>

例如：

  python SARA_complete_system.py article_a.pdf article_a_prime.pdf

输出报告将保存到 ./sara_reports/ 目录下

╔════════════════════════════════════════════════════════════════════╗
    """)
