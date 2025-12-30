# -*- coding: utf-8 -*-
import argparse
from sara_lib import verify_with_a_prime_pdf

def main():
    ap = argparse.ArgumentParser(description="SARA v2 Verify: 输入 package.json + A'_pdf -> 输出 HTML 报告")
    ap.add_argument("--package", required=True, help="prepare 阶段生成的 package.json 路径")
    ap.add_argument("--a_prime_pdf", required=True, help="作者样本 A' 的 PDF 路径（约500字）")
    ap.add_argument("--out_dir", required=True, help="输出目录（可与 prepare 相同）")
    ap.add_argument("--remove_tables", action="store_true", help="尝试剔除表格区域文本（默认不启用）")
    ap.add_argument("--remove_references", action="store_true", help="尝试去除参考文献段（默认不启用）")
    args = ap.parse_args()

    metrics, report = verify_with_a_prime_pdf(
        package_json=args.package,
        a_prime_pdf=args.a_prime_pdf,
        out_dir=args.out_dir,
        remove_tables=args.remove_tables,
        remove_references=args.remove_references
    )

    print("OK. 判定完成：")
    print(f" - verdict: {metrics.verdict}")
    print(f" - confidence: {metrics.overall_confidence:.3f}")
    print(f" - report: {report}")
    print(f" - result.json: {args.out_dir}/result.json")

if __name__ == "__main__":
    main()
