#!/usr/bin/env python3
"""
Apple-style PDF generator for gpt_model4_explained.md
Produces a document that mirrors Apple Developer Documentation aesthetics.
"""

import os
import re
import sys
from datetime import date
from io import BytesIO

# ── ReportLab ─────────────────────────────────────────────────────────────────
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
pt = 1.0  # 1 point = 1 ReportLab unit
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak, Flowable,
    NextPageTemplate,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Pygments (syntax highlighting) ───────────────────────────────────────────
from pygments import lex
from pygments.lexers import PythonLexer, BashLexer, TextLexer, get_lexer_by_name
from pygments.token import (
    Token, Keyword, Name, Literal, String, Number,
    Operator, Punctuation, Comment, Generic, Text as TText,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS  (Apple Design Language)
# ═══════════════════════════════════════════════════════════════════════════════

C_WHITE         = HexColor('#FFFFFF')
C_TEXT          = HexColor('#1d1d1f')       # primary text
C_SECONDARY     = HexColor('#6e6e73')       # secondary / captions
C_BLUE          = HexColor('#0071e3')       # Apple accent blue
C_BLUE_DARK     = HexColor('#0077ed')       # heading blue
C_BLUE_PALE     = HexColor('#e8f1fb')       # inline-code tint
C_DIVIDER       = HexColor('#d2d2d7')       # horizontal rules
C_CODE_BG       = HexColor('#1c1c1e')       # dark code block background
C_CODE_BORDER   = HexColor('#2c2c2e')       # subtle code border
C_CODE_DEFAULT  = HexColor('#f5f5f7')       # default token colour in code
C_CODE_KW       = HexColor('#79b8ff')       # keyword  → blue
C_CODE_BUILTIN  = HexColor('#ffa657')       # built-in → amber
C_CODE_STR      = HexColor('#85e89d')       # string   → green
C_CODE_COMMENT  = HexColor('#6a737d')       # comment  → grey
C_CODE_NUMBER   = HexColor('#f0883e')       # number   → orange
C_CODE_CLASS    = HexColor('#d2a8ff')       # class name → lavender
C_CODE_DECO     = HexColor('#ffd700')       # decorator → gold
C_CODE_SHELL_PS = HexColor('#79b8ff')       # shell prompt $ → blue
C_TABLE_HDR     = HexColor('#f5f5f7')       # table header background
C_TABLE_BORDER  = HexColor('#c7c7cc')       # table border
C_TABLE_ALT     = HexColor('#fafafa')       # alternating row
C_INLINE_CODE   = HexColor('#e8365d')       # inline code text
C_PAGE_NUM      = HexColor('#86868b')       # footer page number
C_COVER_BADGE   = HexColor('#0071e3')       # badge background on cover
C_COVER_BADGE_L = HexColor('#409cff')       # badge subtitle on cover

# ── Page geometry ─────────────────────────────────────────────────────────────
PW, PH = A4                      # 595.27 × 841.89 pt
ML = MR = 2.3 * cm              # left / right margin
MT = 2.8 * cm                   # top margin (header)
MB = 2.3 * cm                   # bottom margin (footer)
CW = PW - ML - MR               # content width ≈ 463 pt


# ═══════════════════════════════════════════════════════════════════════════════
#  FONT REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

# Globals – overwritten if SF fonts register successfully
F_REG  = 'Helvetica'
F_BOLD = 'Helvetica-Bold'
F_ITAL = 'Helvetica-Oblique'
F_MONO = 'Courier'
F_MONO_BOLD = 'Courier-Bold'


def _reg(name, path):
    """Try to register a TTFont; return True on success."""
    if not os.path.exists(path):
        return False
    try:
        pdfmetrics.registerFont(TTFont(name, path))
        return True
    except Exception:
        return False


def register_fonts():
    global F_REG, F_BOLD, F_ITAL, F_MONO, F_MONO_BOLD
    base = '/System/Library/Fonts'

    # SF Pro Text (regular + bold variant from the variable font)
    if _reg('SFNS',      f'{base}/SFNS.ttf') and \
       _reg('SFNSBold',  f'{base}/SFNS.ttf'):   # same file; weight is set in PDF
        F_REG  = 'SFNS'
        F_BOLD = 'SFNSBold'

    if _reg('SFNSItalic', f'{base}/SFNSItalic.ttf'):
        F_ITAL = 'SFNSItalic'

    # SF Mono
    if _reg('SFNSMono', f'{base}/SFNSMono.ttf'):
        F_MONO = 'SFNSMono'
        F_MONO_BOLD = 'SFNSMono'   # same file for both weights


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAGRAPH STYLES
# ═══════════════════════════════════════════════════════════════════════════════

def make_styles() -> dict:
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        # ── headings ──────────────────────────────────────────────────────────
        'h1': S('H1',
            fontName=F_BOLD, fontSize=26, textColor=C_BLUE_DARK,
            leading=32, spaceBefore=30, spaceAfter=4,
        ),
        'h2': S('H2',
            fontName=F_BOLD, fontSize=17, textColor=C_BLUE_DARK,
            leading=22, spaceBefore=24, spaceAfter=4,
        ),
        'h3': S('H3',
            fontName=F_BOLD, fontSize=12, textColor=C_TEXT,
            leading=17, spaceBefore=14, spaceAfter=3,
        ),
        'h4': S('H4',
            fontName=F_BOLD, fontSize=10.5, textColor=C_SECONDARY,
            leading=15, spaceBefore=10, spaceAfter=2,
        ),
        # ── body ──────────────────────────────────────────────────────────────
        'body': S('Body',
            fontName=F_REG, fontSize=10, textColor=C_TEXT,
            leading=16.5, spaceBefore=3, spaceAfter=3,
            alignment=TA_LEFT,
        ),
        'bullet': S('Bullet',
            fontName=F_REG, fontSize=10, textColor=C_TEXT,
            leading=16, spaceBefore=2, spaceAfter=2,
            leftIndent=18, firstLineIndent=0,
        ),
        'sub_bullet': S('SubBullet',
            fontName=F_REG, fontSize=10, textColor=C_TEXT,
            leading=16, spaceBefore=1, spaceAfter=1,
            leftIndent=36, firstLineIndent=0,
        ),
        # ── tables ────────────────────────────────────────────────────────────
        'table_hdr': S('TblHdr',
            fontName=F_BOLD, fontSize=8.5, textColor=C_TEXT,
            leading=13, alignment=TA_LEFT,
        ),
        'table_cell': S('TblCell',
            fontName=F_REG, fontSize=8.5, textColor=C_TEXT,
            leading=13, alignment=TA_LEFT,
        ),
        # ── misc ──────────────────────────────────────────────────────────────
        'caption': S('Caption',
            fontName=F_REG, fontSize=8, textColor=C_SECONDARY,
            leading=12, spaceAfter=6,
        ),
        'cover_title': S('CoverTitle',
            fontName=F_BOLD, fontSize=38, textColor=C_BLUE,
            leading=46, alignment=TA_CENTER, spaceAfter=10,
        ),
        'cover_sub': S('CoverSub',
            fontName=F_REG, fontSize=13.5, textColor=C_TEXT,
            leading=20, alignment=TA_CENTER, spaceAfter=4,
        ),
        'cover_meta': S('CoverMeta',
            fontName=F_REG, fontSize=9, textColor=C_SECONDARY,
            leading=14, alignment=TA_CENTER,
        ),
        'cover_badge_title': S('CBT',
            fontName=F_BOLD, fontSize=10, textColor=C_WHITE,
            leading=14, alignment=TA_CENTER,
        ),
        'cover_badge_sub': S('CBS',
            fontName=F_REG, fontSize=8, textColor=C_COVER_BADGE_L,
            leading=12, alignment=TA_CENTER,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTAX HIGHLIGHTING  (Pygments → colour map)
# ═══════════════════════════════════════════════════════════════════════════════

_TOKEN_COLOUR = {
    # Keywords
    Token.Keyword:                          C_CODE_KW,
    Token.Keyword.Constant:                 C_CODE_KW,
    Token.Keyword.Declaration:              C_CODE_KW,
    Token.Keyword.Namespace:                C_CODE_KW,
    Token.Keyword.Type:                     C_CODE_KW,
    # Names
    Token.Name.Builtin:                     C_CODE_BUILTIN,
    Token.Name.Builtin.Pseudo:              C_CODE_KW,
    Token.Name.Class:                       C_CODE_CLASS,
    Token.Name.Decorator:                   C_CODE_DECO,
    Token.Name.Exception:                   C_CODE_CLASS,
    Token.Name.Function:                    C_CODE_DEFAULT,
    # Literals
    Token.Literal.String:                   C_CODE_STR,
    Token.Literal.String.Doc:               C_CODE_COMMENT,
    Token.Literal.Number:                   C_CODE_NUMBER,
    Token.Literal.Number.Integer:           C_CODE_NUMBER,
    Token.Literal.Number.Float:             C_CODE_NUMBER,
    Token.Literal.Number.Hex:               C_CODE_NUMBER,
    # Comments
    Token.Comment:                          C_CODE_COMMENT,
    Token.Comment.Single:                   C_CODE_COMMENT,
    Token.Comment.Multiline:                C_CODE_COMMENT,
    # Operators & punctuation
    Token.Operator:                         C_CODE_DEFAULT,
    Token.Punctuation:                      C_CODE_DEFAULT,
    # Error
    Token.Error:                            C_CODE_DEFAULT,
}


def _token_color(ttype) -> HexColor:
    """Walk up the token hierarchy to find the best colour match."""
    while ttype is not Token:
        if ttype in _TOKEN_COLOUR:
            return _TOKEN_COLOUR[ttype]
        ttype = ttype.parent
    return C_CODE_DEFAULT


def get_lexer(lang: str):
    lang = (lang or '').lower().strip()
    if lang in ('python', 'py'):
        return PythonLexer(stripnl=False)
    if lang in ('bash', 'sh', 'shell'):
        return BashLexer(stripnl=False)
    try:
        return get_lexer_by_name(lang, stripnl=False)
    except Exception:
        return TextLexer(stripnl=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  CODE BLOCK FLOWABLE
# ═══════════════════════════════════════════════════════════════════════════════

class CodeBlock(Flowable):
    """Dark-background code block with Pygments syntax highlighting."""

    PAD_H    = 18     # horizontal inner padding (pt)
    PAD_V    = 14     # vertical inner padding (pt)
    LINE_H   = 13.5   # height per code line (pt)
    FONT_SZ  = 8.2    # monospace font size (pt)
    RADIUS   = 8      # corner radius (pt)
    TAB_W    = 4      # tab → spaces

    def __init__(self, code: str, lang: str = 'python', avail_width: float = CW):
        super().__init__()
        self.code  = code.expandtabs(self.TAB_W).rstrip('\n')
        self.lang  = lang
        self._aW   = avail_width
        self._lines = self.code.split('\n')
        self._h    = len(self._lines) * self.LINE_H + 2 * self.PAD_V

        # Pre-tokenise once per instance
        lexer = get_lexer(lang)
        self._tokens = list(lex(self.code, lexer))

    # ── Flowable protocol ────────────────────────────────────────────────────

    def wrap(self, availWidth, availHeight):
        self._aW = availWidth
        return availWidth, self._h

    def split(self, availWidth, availHeight):
        """Split tall code blocks across pages.

        ReportLab calls this when wrap() height > availHeight.
        Must return ≥2 items, where item[0] fits in availHeight.
        """
        if availHeight >= self._h:
            return [self]   # fits entirely — shouldn't be called, but safe

        usable = availHeight - 2 * self.PAD_V
        n_fit  = int(usable / self.LINE_H)   # lines that genuinely fit

        if n_fit <= 0 or n_fit >= len(self._lines):
            # Nothing fits (tiny gap) OR padding alone makes it overflow:
            # push the whole block to the next page via a filler spacer.
            return [Spacer(1, availHeight), self]

        top_code    = '\n'.join(self._lines[:n_fit])
        bottom_code = '\n'.join(self._lines[n_fit:])
        return [
            CodeBlock(top_code,    lang=self.lang, avail_width=availWidth),
            CodeBlock(bottom_code, lang=self.lang, avail_width=availWidth),
        ]

    def draw(self):
        c   = self.canv
        w   = self._aW
        h   = self._h

        # ── Background rounded rect ──────────────────────────────────────
        c.setFillColor(C_CODE_BG)
        c.setStrokeColor(C_CODE_BORDER)
        c.setLineWidth(0.5)
        c.roundRect(0, 0, w, h, self.RADIUS, stroke=1, fill=1)

        # ── Language label (top-right, subtle) ───────────────────────────
        label = self.lang.lower() if self.lang not in ('', 'text') else ''
        if label:
            c.setFont(F_MONO, 7)
            c.setFillColor(HexColor('#555558'))
            c.drawRightString(w - self.PAD_H, h - self.PAD_V + 3, label)

        # ── Render tokens line-by-line ───────────────────────────────────
        x0 = self.PAD_H
        y  = h - self.PAD_V - self.FONT_SZ   # baseline of first line

        cur_x = x0
        cur_y = y

        for ttype, value in self._tokens:
            colour = _token_color(ttype)
            segments = value.split('\n')

            for seg_idx, seg in enumerate(segments):
                if seg:
                    c.setFont(F_MONO, self.FONT_SZ)
                    c.setFillColor(colour)
                    c.drawString(cur_x, cur_y, seg)
                    cur_x += c.stringWidth(seg, F_MONO, self.FONT_SZ)

                if seg_idx < len(segments) - 1:
                    # newline → next line
                    cur_y -= self.LINE_H
                    cur_x  = x0

    def __repr__(self):
        return f'<CodeBlock lang={self.lang!r} lines={len(self._lines)}>'


# ═══════════════════════════════════════════════════════════════════════════════
#  INLINE MARKUP  (markdown → ReportLab XML)
# ═══════════════════════════════════════════════════════════════════════════════

_MONO_TAG_OPEN  = f'<font name="{F_MONO}" size="8.5" color="#e8365d">'
_MONO_TAG_CLOSE = '</font>'


def _escape_xml(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def md_inline(text: str) -> str:
    """Convert **bold**, *italic*, `code`, [link]() inline markdown to RL XML.

    Uses a placeholder strategy so that inline-code spans are extracted first
    and restored after bold/italic substitution, preventing tag interleaving.
    """
    # 1. Strip markdown links [text](url) → just text (before escaping)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 2. Pull out inline code spans before escaping so their content is safe
    placeholders = {}
    ph_idx = [0]

    def stash_code(m):
        key = f'\x00CODE{ph_idx[0]}\x00'
        ph_idx[0] += 1
        inner = _escape_xml(m.group(1))
        placeholders[key] = (
            f'<font name="{F_MONO}" size="8.5" color="#e8365d">{inner}</font>'
        )
        return key

    text = re.sub(r'`([^`]+)`', stash_code, text)

    # 3. Escape XML special chars in the remaining prose
    text = _escape_xml(text)

    # 4. Bold **...** (must not contain newlines)
    text = re.sub(r'\*\*([^*\n]+?)\*\*', r'<b>\1</b>', text)

    # 5. Italic *...*  (single star, not adjacent to another star)
    text = re.sub(r'(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)

    # 6. Restore code placeholders
    for key, val in placeholders.items():
        # key was xml-escaped in step 3 — need to match escaped version
        text = text.replace(_escape_xml(key), val)

    return text


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_table(md_rows: list, styles: dict) -> Table:
    """Convert a list of raw markdown table rows into a styled ReportLab Table."""
    data = []
    for row in md_rows:
        cells = [c.strip() for c in row.strip('|').split('|')]
        data.append(cells)

    if not data:
        return Spacer(1, 0)

    n_cols = max(len(r) for r in data)
    data   = [r + [''] * (n_cols - len(r)) for r in data]

    # Header row
    header = [Paragraph(md_inline(c), styles['table_hdr']) for c in data[0]]
    body   = [
        [Paragraph(md_inline(c), styles['table_cell']) for c in row]
        for row in data[1:]
    ]
    table_data = [header] + body

    # Column widths — equal, but cap at CW
    col_w = min(CW / n_cols, 160)
    tbl_w = col_w * n_cols
    col_widths = [col_w] * n_cols

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1,
                hAlign='LEFT')

    alt_cmds = [
        ('BACKGROUND', (0, r), (-1, r), C_TABLE_ALT)
        for r in range(2, len(table_data), 2)
    ]

    tbl.setStyle(TableStyle([
        # Header
        ('BACKGROUND',    (0, 0), (-1, 0),  C_TABLE_HDR),
        ('FONTNAME',      (0, 0), (-1, 0),  F_BOLD),
        ('FONTSIZE',      (0, 0), (-1, 0),  8.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0),  7),
        ('TOPPADDING',    (0, 0), (-1, 0),  7),
        # Body
        ('FONTNAME',      (0, 1), (-1, -1), F_REG),
        ('FONTSIZE',      (0, 1), (-1, -1), 8.5),
        ('TOPPADDING',    (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        # Borders
        ('LINEBELOW',     (0, 0), (-1, 0),  1.2, C_BLUE),
        ('LINEBELOW',     (0, 1), (-1, -2), 0.3, C_DIVIDER),
        ('BOX',           (0, 0), (-1, -1), 0.5, C_TABLE_BORDER),
        # All cells
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        *alt_cmds,
    ]))
    return tbl


# ═══════════════════════════════════════════════════════════════════════════════
#  MARKDOWN PARSER  →  list of ReportLab flowables
# ═══════════════════════════════════════════════════════════════════════════════

_RE_HR     = re.compile(r'^(-{3,}|_{3,})$')
_RE_BULLET = re.compile(r'^(\s*)[-*+] (.*)')
_RE_NUMLI  = re.compile(r'^(\s*)(\d+)\. (.*)')
_RE_TABLE  = re.compile(r'^\|')
_RE_SEPRTR = re.compile(r'^\|[-| :]+\|$')


def parse_markdown(md_text: str, styles: dict) -> list:
    flowables = []
    lines = md_text.split('\n')
    i = 0

    def add(f):
        flowables.append(f)

    def p(text, sk='body'):
        t = md_inline(text)
        if t.strip():
            add(Paragraph(t, styles[sk]))

    while i < len(lines):
        line = lines[i]

        # ── Fenced code block  ``` ... ``` ───────────────────────────────────
        if line.startswith('```'):
            lang = line[3:].strip() or 'text'
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            code = '\n'.join(code_lines)
            add(Spacer(1, 5))
            add(CodeBlock(code, lang=lang, avail_width=CW))
            add(Spacer(1, 9))
            i += 1
            continue

        # ── Horizontal rule ───────────────────────────────────────────────────
        if _RE_HR.match(line.strip()):
            add(Spacer(1, 5))
            add(HRFlowable(width='100%', thickness=0.4, color=C_DIVIDER))
            add(Spacer(1, 5))
            i += 1
            continue

        # ── Headings ──────────────────────────────────────────────────────────
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            add(Spacer(1, 6))
            add(Paragraph(md_inline(text), styles['h1']))
            add(HRFlowable(width='100%', thickness=1.5,
                           color=C_BLUE, spaceAfter=10))
            i += 1
            continue

        if line.startswith('## ') and not line.startswith('### '):
            text = line[3:].strip()
            add(Paragraph(md_inline(text), styles['h2']))
            add(HRFlowable(width='100%', thickness=0.4,
                           color=C_DIVIDER, spaceAfter=5))
            i += 1
            continue

        if line.startswith('### ') and not line.startswith('#### '):
            add(Paragraph(md_inline(line[4:].strip()), styles['h3']))
            i += 1
            continue

        if line.startswith('#### '):
            add(Paragraph(md_inline(line[5:].strip()), styles['h4']))
            i += 1
            continue

        # ── Markdown table ────────────────────────────────────────────────────
        if _RE_TABLE.match(line):
            rows = []
            while i < len(lines) and _RE_TABLE.match(lines[i]):
                if not _RE_SEPRTR.match(lines[i]):
                    rows.append(lines[i])
                i += 1
            if rows:
                add(Spacer(1, 5))
                add(build_table(rows, styles))
                add(Spacer(1, 9))
            continue

        # ── Bullet list ───────────────────────────────────────────────────────
        if _RE_BULLET.match(line):
            while i < len(lines) and _RE_BULLET.match(lines[i]):
                m      = _RE_BULLET.match(lines[i])
                indent = len(m.group(1))
                text   = m.group(2)
                sk     = 'sub_bullet' if indent >= 2 else 'bullet'
                bullet = '◦' if indent >= 2 else '•'
                add(Paragraph(f'{bullet} {md_inline(text)}', styles[sk]))
                i += 1
            add(Spacer(1, 3))
            continue

        # ── Numbered list ─────────────────────────────────────────────────────
        if _RE_NUMLI.match(line):
            while i < len(lines) and _RE_NUMLI.match(lines[i]):
                m    = _RE_NUMLI.match(lines[i])
                num  = m.group(2)
                text = m.group(3)
                add(Paragraph(f'{num}. {md_inline(text)}', styles['bullet']))
                i += 1
            add(Spacer(1, 3))
            continue

        # ── Empty line ────────────────────────────────────────────────────────
        if not line.strip():
            add(Spacer(1, 4))
            i += 1
            continue

        # ── Normal paragraph (collect continuation lines) ─────────────────────
        para_lines = [line]
        i += 1
        while (i < len(lines)
               and lines[i].strip()
               and not lines[i].startswith('#')
               and not lines[i].startswith('```')
               and not _RE_TABLE.match(lines[i])
               and not _RE_HR.match(lines[i].strip())
               and not _RE_BULLET.match(lines[i])
               and not _RE_NUMLI.match(lines[i])):
            para_lines.append(lines[i])
            i += 1

        full = ' '.join(para_lines)
        add(Paragraph(md_inline(full), styles['body']))

    return flowables


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE TEMPLATE  (header + footer)
# ═══════════════════════════════════════════════════════════════════════════════

DOC_TITLE = 'gpt_model4.py — Technical Reference'
DOC_LABEL = 'GPT-4 Style Architecture'


def on_page(canvas, doc):
    canvas.saveState()

    # ── Header ───────────────────────────────────────────────────────────────
    y_hdr = PH - MT + 10

    canvas.setFont(F_REG, 8)
    canvas.setFillColor(C_SECONDARY)
    canvas.drawString(ML, y_hdr, DOC_TITLE)

    canvas.setFillColor(C_BLUE)
    canvas.drawRightString(PW - MR, y_hdr, DOC_LABEL)

    canvas.setStrokeColor(C_BLUE)
    canvas.setLineWidth(0.6)
    canvas.line(ML, y_hdr - 4, PW - MR, y_hdr - 4)

    # ── Footer ────────────────────────────────────────────────────────────────
    y_ftr = MB - 10

    canvas.setStrokeColor(C_DIVIDER)
    canvas.setLineWidth(0.3)
    canvas.line(ML, y_ftr + 5, PW - MR, y_ftr + 5)

    canvas.setFont(F_REG, 7.5)
    canvas.setFillColor(C_PAGE_NUM)
    canvas.drawCentredString(PW / 2, y_ftr - 4, f'{doc.page}')

    canvas.restoreState()


def on_first_page(canvas, doc):
    """Cover page — no header/footer."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  COVER PAGE FLOWABLES
# ═══════════════════════════════════════════════════════════════════════════════

def make_cover(styles: dict) -> list:
    els = []

    els.append(Spacer(1, 90))

    # ── Main title ───────────────────────────────────────────────────────────
    els.append(Paragraph('gpt_model4.py', styles['cover_title']))

    # ── Subtitle ─────────────────────────────────────────────────────────────
    els.append(Paragraph(
        'GPT-4 Style Language Model — Line-by-Line Technical Reference',
        styles['cover_sub'],
    ))

    els.append(Spacer(1, 20))
    els.append(HRFlowable(width='55%', thickness=2, color=C_BLUE, hAlign='CENTER'))
    els.append(Spacer(1, 24))

    # ── Four architecture badges ──────────────────────────────────────────────
    badges = [
        ('RMSNorm',   'replaces LayerNorm'),
        ('RoPE',      'replaces pos_emb'),
        ('GQA',       'replaces MHA'),
        ('SwiGLU',    'replaces GELU MLP'),
    ]

    def badge_cell(title, sub):
        inner = Table(
            [[Paragraph(title, styles['cover_badge_title'])],
             [Paragraph(sub,   styles['cover_badge_sub'])]],
            colWidths=[96],
        )
        inner.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), C_COVER_BADGE),
            ('TOPPADDING',    (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
            ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return inner

    badge_row = Table(
        [[badge_cell(t, s) for t, s in badges]],
        colWidths=[104] * 4,
        hAlign='CENTER',
    )
    badge_row.setStyle(TableStyle([
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 4),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
    ]))
    els.append(badge_row)

    els.append(Spacer(1, 36))

    # ── Meta line ─────────────────────────────────────────────────────────────
    els.append(Paragraph(
        f'PyTorch · LLaMA-class Architecture · {date.today().strftime("%B %Y")}',
        styles['cover_meta'],
    ))

    els.append(NextPageTemplate('Content'))
    els.append(PageBreak())
    return els


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN BUILD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_pdf(md_path: str, pdf_path: str) -> None:
    register_fonts()
    styles  = make_styles()

    # Read source
    with open(md_path, 'r', encoding='utf-8') as fh:
        md_text = fh.read()

    # Parse to flowables
    body = parse_markdown(md_text, styles)

    # Document
    doc = BaseDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT,  bottomMargin=MB,
        title='gpt_model4.py — Technical Reference',
        author='LLM Project',
        subject='GPT-4 Style Architecture Documentation',
    )

    # Page templates
    content_frame = Frame(ML, MB, CW, PH - MT - MB,
                          leftPadding=0, rightPadding=0,
                          topPadding=0,  bottomPadding=0,
                          id='main')
    cover_frame   = Frame(ML, MB, CW, PH - MB - 1*cm,
                          leftPadding=0, rightPadding=0,
                          topPadding=0,  bottomPadding=0,
                          id='cover')

    doc.addPageTemplates([
        PageTemplate(id='Cover',   frames=[cover_frame],   onPage=on_first_page),
        PageTemplate(id='Content', frames=[content_frame], onPage=on_page),
    ])

    story = make_cover(styles) + [
        # Switch to content template after cover
        # (PageBreak from cover already triggers template change via NextPageTemplate)
    ] + body

    doc.build(story)
    print(f'✓  PDF saved → {pdf_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    SRC = os.path.join(os.path.dirname(__file__), 'gpt_model4_explained.md')
    DST = os.path.join(os.path.dirname(__file__), 'gpt_model4_explained.pdf')
    build_pdf(SRC, DST)
