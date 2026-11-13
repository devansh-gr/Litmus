"""Builds litmus_note.docx in IEEE conference layout (letter, Times 10pt, two columns,
full-width title + figure). Equations are native Word OMML. Run: python make_docx.py"""
import re
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt

M = nsdecls("m", "w")

# ---------- OMML builders ----------
def r(t, roman=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if roman else ""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<m:r>{sty}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr><m:t xml:space="preserve">{t}</m:t></m:r>'
def sub(b, s): return f"<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>"
def frac(n, d): return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"
def bar(e): return f'<m:acc><m:accPr><m:chr m:val="̄"/></m:accPr><m:e>{e}</m:e></m:acc>'
def nary(c, lo, hi, e):
    return (f'<m:nary><m:naryPr><m:chr m:val="{c}"/></m:naryPr><m:sub>{lo}</m:sub>'
            f"<m:sup>{hi}</m:sup><m:e>{e}</m:e></m:nary>")
def omath(x): return parse_xml(f"<m:oMath {M}>{x}</m:oMath>")

lb = bar(r("ℓ"))
EQ = {
    1: r("s(y ∣ x)=") + frac(r("1"), r("|y|")) + nary("∑", r("t=1"), r("|y|"),
        r("log", True) + r(" ") + sub(r("p"), r("θ")) + r("(") + sub(r("y"), r("t")) +
        r(" ∣ x, ") + sub(r("y"), r("<t")) + r(")")),
    2: sub(r("M"), r("tok", True)) + r("=2 L ") + sub(r("H"), r("kv")) + r(" ") + sub(r("d"), r("h")) + r(" b"),
    3: r("S(K)=") + frac(r("K(P+") + lb + r(")"), r("P+K(") + lb + r("−1)")),
    4: sub(r("t"), r("dec", True)) + r("(n) ≥ ") + frac(r("nW"), r("B")),
}

# ---------- document setup ----------
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"; st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
pf = st.paragraph_format; pf.space_before = pf.space_after = Pt(0); pf.line_spacing = 1.0

s0 = doc.sections[0]
s0.page_width, s0.page_height = Inches(8.5), Inches(11)
s0.left_margin = s0.right_margin = Inches(0.625)
s0.top_margin, s0.bottom_margin = Inches(0.5), Inches(0.6)
COLW = 3.5

def para(align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, after=0, before=0):
    p = doc.add_paragraph(); p.alignment = align
    p.paragraph_format.first_line_indent = Inches(0.14) if indent else None
    p.paragraph_format.space_after = Pt(after); p.paragraph_format.space_before = Pt(before)
    return p

SYM = {r"\mathcal{Y}": "𝒴", r"\bar\ell": "ℓ̄", r"\approx": "≈", r"\times": "×", r"\in": "∈",
       r"\ge": "≥", r"\theta": "θ"}

def rich(p, text, size=None, bold=False):
    """Plain text with $math$ (italic letters, _{sub}) and **bold** spans."""
    for i, part in enumerate(re.split(r"(\$[^$]+\$)", text)):
        if part.startswith("$"):
            m = part[1:-1]
            for k, v in SYM.items(): m = m.replace(k, v)
            for tok in re.findall(r"_\{[^}]+\}|_.|[A-Za-z𝒴ℓ]̄?|[^A-Za-z_]", m):
                run = p.add_run(tok[2:-1] if tok.startswith("_{") else tok.lstrip("_"))
                if tok.startswith("_"): run.font.subscript = True
                elif tok[0].isalpha() or tok[0] in "𝒴ℓ": run.italic = True
                run.bold = bold
                if size: run.font.size = Pt(size)
        else:
            for j, seg in enumerate(part.split("**")):
                run = p.add_run(seg); run.bold = bold or j % 2 == 1
                if size: run.font.size = Pt(size)
    return p

def heading(num, title):
    p = para(WD_ALIGN_PARAGRAPH.CENTER, False, after=2, before=4)
    run = p.add_run(f"{num}. {title}"); run.font.small_caps = True

def equation(n):
    p = para(WD_ALIGN_PARAGRAPH.LEFT, False, after=2, before=2)
    ts = p.paragraph_format.tab_stops
    ts.add_tab_stop(Inches(COLW / 2), WD_TAB_ALIGNMENT.CENTER)
    ts.add_tab_stop(Inches(COLW), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t"); p._p.append(omath(EQ[n])); p.add_run(f"\t({n})")

# ---------- title block (one column) ----------
p = para(WD_ALIGN_PARAGRAPH.CENTER, False, after=6)
run = p.add_run("Litmus: KV-Cache Reuse and the Accuracy-Latency Frontier of On-Device Manipulation Detection")
run.font.size = Pt(20)
p = para(WD_ALIGN_PARAGRAPH.CENTER, False); p.add_run("Devansh Gaur").font.size = Pt(11)
p = para(WD_ALIGN_PARAGRAPH.CENTER, False, after=8)
p.add_run("University of Waterloo, Waterloo, ON, Canada").italic = True

p = para(WD_ALIGN_PARAGRAPH.CENTER, False)
p.add_run().add_picture("fig_main.png", width=Inches(7.16))
p = para(WD_ALIGN_PARAGRAPH.JUSTIFY, False, after=6)
rich(p, "Fig. 1. (a) Accuracy vs. latency on Apple M4 Pro, Wilson 95% intervals (*R1: balanced $N$=42 "
        "subset). (b) fp16 KV cache from each config.json; dotted is a 14B multi-head counterfactual. "
        "(c) Eq. (3) vs. measured p50 speedup. (d) Eq. (4) decode bound.", size=8)

# ---------- two-column body ----------
sec = doc.add_section(WD_SECTION.CONTINUOUS)
cols = sec._sectPr.xpath("./w:cols")
c = cols[0] if cols else sec._sectPr.makeelement(qn("w:cols"), {})
c.set(qn("w:num"), "2"); c.set(qn("w:space"), str(int(0.25 * 1440)))
if not cols: sec._sectPr.append(c)

p = para(indent=False, after=4)
a = p.add_run("Abstract: "); a.bold = a.italic = True; a.font.size = Pt(9)
rich(p, "Litmus labels a text selection with one of 14 persuasion techniques or none by scoring "
        "candidate labels under a local 4-bit LLM. This note analyzes KV-cache memory, prefix KV reuse, "
        "and the decode bandwidth bound. Held-out top-1 accuracy rose from 62.7% (3B) to 82.3% (14B, "
        "few-shot); prefix reuse cut median latency 9.0×.", size=9, bold=True)

heading("I", "Scoring as Constrained Decoding")
rich(para(), "The detector never generates text. For input $x$ and each label $y \\in \\mathcal{Y}$ "
             "($K$=|$\\mathcal{Y}$|=15) it computes")
equation(1)
rich(para(indent=False), "and returns argmax$_{y}$ $s$. Length normalization removes the bias toward short "
     "labels (1 to 7 tokens). A two-label gate first screens benign text (PR-AUC 0.93). Ranked scores give "
     "top-2 accuracy of 73.0% against 64.0% top-1 on the 3B, so most errors name a co-present technique.")

heading("II", "KV-Cache Memory")
rich(para(), "Each attention layer caches one key and one value per token:")
equation(2)
rich(para(indent=False), "All three models use grouped-query attention [1] with $H_{kv}$=8, $d_{h}$=128, giving "
     "112, 128 and 192 KiB per token (3B, 8B, 14B, fp16). The 1446-token few-shot prompt occupies 271 MiB "
     "on the 14B, against 1.32 GiB under full multi-head attention with $H_{kv}$=40 (Fig. 1b). Beside 7.7 GB "
     "of weights the cache is minor, so RAM did not limit the 14B on a 24 GB machine.")

heading("III", "Prefix KV Reuse")
rich(para(), "All labels share a $P$-token prefix. The deployed scorer prefills it once, appends each label "
     "to the cache, reads its log-probabilities, then trims the cache back to length $P$. Verdicts are "
     "unchanged. Relative to $K$ independent passes, forwarded tokens fall by")
equation(3)
rich(para(indent=False), "At $P$=1446 the scorer forwards 1493 tokens instead of 21,752 (14.6×). At "
     "measurement time ($K$=12, $P \\approx$ 350) Eq. (3) predicted 11.0×; median latency fell from 4014 to "
     "446 ms (9.0×, Fig. 1c). The gap is fixed per-call overhead and label tokens attending over the full prefix.")

heading("IV", "Prefill Versus Decode")
rich(para(), "Prefill is parallel and compute-bound. Decode emits one token per step and streams all "
     "weights each step:")
equation(4)
rich(para(indent=False), "With $W$=7.7 GB and $B$=273 GB/s, the 14B needs at least 28 ms per token. Log-prob "
     "scoring decodes zero tokens. DeepSeek-R1-Distill-14B [2] must decode a chain of thought first: it "
     "averaged 35 s per scan, which bounds its mean chain at about 1240 tokens (Fig. 1d), and reached only "
     "76.2% on a balanced subset.")

heading("V", "Results and Discussion")
rich(para(), "Accuracy rose monotonically with size (Table I), so 62.7% was a 3B capacity limit. Few-shot "
     "exemplars helped the 14B by three points but not the 3B. LoRA [3] on 503 imbalanced examples did not "
     "beat few-shot, and self-consistency voting [4] gained nothing on cleaned labels. The 3B serves the "
     "0.6 s shortcut and the 14B is an opt-in deep scan: the split is set by compute and bandwidth, not memory.")

# ---------- Table I ----------
p = para(WD_ALIGN_PARAGRAPH.CENTER, False, before=6)
p.add_run("TABLE I").font.size = Pt(8)
p = para(WD_ALIGN_PARAGRAPH.CENTER, False, after=2)
run = p.add_run("Held-out Benchmark (N=300), All On-Device"); run.font.small_caps = True; run.font.size = Pt(8)
rows = [("Configuration", "Top-1 (%)", "MCC", "Latency (s)"),
        ("3B, few-shot (shipped)", "62.7", "0.55", "0.6"),
        ("   + gate fix, clean set", "66.9", "n/a", "0.6"),
        ("8B, zero-shot", "69.0", "0.634", "7.4"),
        ("14B, zero-shot", "79.3", "0.752", "12.7"),
        ("14B, few-shot", "82.3", "0.787", "7 to 13"),
        ("14B, LoRA (N=123)", "78.9", "n/a", "n/a")]
t = doc.add_table(rows=len(rows), cols=4); t.alignment = WD_TABLE_ALIGNMENT.CENTER
t.autofit = False
W = [1.5, 0.65, 0.5, 0.75]
tblPr = t._tbl.tblPr
tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), W): gc.set(qn("w:w"), str(int(w * 1440)))
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        cell = t.cell(i, j); cell.width = Inches(W[j])
        cp = cell.paragraphs[0]; cp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
        run = cp.add_run(val); run.font.size = Pt(8); run.bold = i == 5
def border(cell, **sides):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f"<w:tcBorders {nsdecls('w')}>" + "".join(
        f'<w:{s} w:val="single" w:sz="{sz}" w:color="000000"/>' for s, sz in sides.items()) + "</w:tcBorders>"))
for j in range(4):
    border(t.cell(0, j), top=8, bottom=4); border(t.cell(len(rows) - 1, j), bottom=8)

# ---------- references ----------
p = para(WD_ALIGN_PARAGRAPH.CENTER, False, after=2, before=6)
p.add_run("References").font.small_caps = True
refs = [
    "J. Ainslie et al., “GQA: Training generalized multi-query transformer models from multi-head checkpoints,” in Proc. EMNLP, 2023.",
    "DeepSeek-AI, “DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning,” arXiv:2501.12948, 2025.",
    "E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022.",
    "X. Wang et al., “Self-consistency improves chain of thought reasoning in language models,” in Proc. ICLR, 2023.",
]
for i, ref in enumerate(refs, 1):
    p = para(WD_ALIGN_PARAGRAPH.JUSTIFY, False)
    p.paragraph_format.left_indent = Inches(0.25); p.paragraph_format.first_line_indent = Inches(-0.25)
    p.add_run(f"[{i}]\t{ref}").font.size = Pt(8)

doc.save("litmus_note.docx")
print("saved")
