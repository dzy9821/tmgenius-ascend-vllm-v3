from __future__ import annotations

import re

_PROVINCES = r'京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁'

# 车牌号位字符：半角字母数字 + 全角字母数字 + 中文数字（含口语变体）
_ALNUM = r'0-9A-Za-z０-９Ａ-Ｚａ-ｚ'
_CN_DIGIT = r'零〇一二三四五六七八九幺洞两'
_SEP = r'[·．.·\s\-]*'

_PLATE_PATTERN = re.compile(
    rf'[{_PROVINCES}]'
    rf'[A-Za-zＡ-Ｚａ-ｚ]'
    rf'(?:{_SEP}[{_ALNUM}{_CN_DIGIT}]){{5,6}}'
)

# 全角字母数字 → 半角
_FULLWIDTH_TABLE = str.maketrans(
    {c: chr(ord(c) - 0xFEE0) for c in 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
     'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
     '０１２３４５６７８９'}
)

# 中文数字 → 阿拉伯数字
_CN_DIGIT_MAP = str.maketrans({
    '零': '0', '〇': '0',
    '一': '1', '幺': '1',
    '二': '2', '两': '2',
    '三': '3',
    '四': '4',
    '五': '5',
    '六': '6',
    '七': '7',
    '八': '8',
    '九': '9',
    '洞': '0',
})


def normalize_license_plates(text: str) -> str:
    """扫描文本中的车牌号，统一大写、去符号、中文数字转阿拉伯数字、全角转半角。"""
    if not text:
        return text

    def _clean(m: re.Match) -> str:
        raw = m.group(0)
        raw = raw.translate(_FULLWIDTH_TABLE)
        raw = raw.translate(_CN_DIGIT_MAP)
        raw = re.sub(r'[·．.·\s\-]', '', raw)
        return raw.upper()

    return _PLATE_PATTERN.sub(_clean, text)
