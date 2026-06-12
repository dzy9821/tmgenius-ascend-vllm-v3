"""
ITN (Inverse Text Normalization) 逆文本规范化测试脚本。

将中文口语化数字/符号转换为书面形式，例如：
  "幺幺零" → "110"
  "一百二十三" → "123"
  "二零二三年五月八号" → "2023年05月08日"
  "三点一四" → "3.14"

用法:
  source .venv/bin/activate
  python test/itn_inverse_normalize.py                      # 内置测试用例
  python test/itn_inverse_normalize.py "幺幺零"              # 单个输入
  python test/itn_inverse_normalize.py --interactive         # 交互模式
"""

import argparse
import os
import sys

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def load_itn_processor():
    """加载 ITNProcessor（使用项目内置的 itn_wrapper）。"""
    from weights.itn.itn_wrapper import ITNProcessor

    model_path = os.path.join(_PROJECT_ROOT, "weights", "fst_itn_zh")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"ITN model directory not found: {model_path}")
    return ITNProcessor(model_path=model_path)


# 内置测试用例 — 展示 ITN 将中文口语转换为书面形式
BUILTIN_TEST_CASES = [
    # 数字
    ("幺幺零", "110"),
    ("幺二零", "120"),
    ("一百二十三", "123"),
    ("五万六千七百八十九", "56789"),
    ("三点一四", "3.14"),
    ("三点一四一五九", "3.14159"),
    ("百分之九十", "90%"),
    ("百分之五十", "50%"),
    # 日期
    ("二零二三年五月八号", "2023年05月08日"),
    ("二零二五年十二月二十五号", "2025年12月25日"),
    # 时间
    ("八点二十", "8:20"),
    ("十点零五分", "10:05"),
    # 分数
    ("三分之一", "1/3"),
    # 数学
    ("一加一等于二", "1+1=2"),
    # 车牌
    ("京A一二三四五", "京A12345"),
]


def run_builtin_tests(processor):
    """运行内置测试用例并输出结果。"""
    max_len = max(len(case[0]) for case in BUILTIN_TEST_CASES) + 4
    passed = 0
    failed = 0

    print(f"{'输入':<{max_len}} {'期望输出':<{max_len}} {'实际输出':<{max_len}} 结果")
    print("-" * (max_len * 3 + 10))

    for input_text, expected in BUILTIN_TEST_CASES:
        actual = processor.process(input_text)
        ok = actual == expected
        if ok:
            passed += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"

        print(f"{input_text:<{max_len}} {expected:<{max_len}} {actual:<{max_len}} {status}")

    print("-" * (max_len * 3 + 10))
    print(f"通过: {passed}, 失败: {failed}")


def interactive_mode(processor):
    """交互模式：用户反复输入文本并查看 ITN 结果。"""
    print("ITN 交互模式 — 输入文本查看逆规范化结果，输入 q/quit 退出")
    print("提示: 可以输入 '幺幺零'、'一百二十三'、'下午三点' 等")
    print()
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            continue
        result = processor.process(text)
        print(f"  → {result}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="ITN 逆文本规范化测试工具"
    )
    parser.add_argument(
        "text", nargs="?", default=None,
        help="要进行 ITN 处理的文本（不提供则运行内置测试用例）"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="交互模式"
    )
    parser.add_argument(
        "--model-path", default=None,
        help="ITN 模型目录（默认: weights/fst_itn_zh）"
    )
    args = parser.parse_args()

    model_path = args.model_path or os.path.join(_PROJECT_ROOT, "weights", "fst_itn_zh")

    print(f"Loading ITN model from: {model_path}")
    processor = load_itn_processor()
    print("ITN processor ready.\n")

    if args.interactive:
        interactive_mode(processor)
    elif args.text:
        result = processor.process(args.text)
        print(f"输入: {args.text}")
        print(f"输出: {result}")
    else:
        run_builtin_tests(processor)


if __name__ == "__main__":
    main()
