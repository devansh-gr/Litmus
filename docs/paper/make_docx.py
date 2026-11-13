"""Builds litmus_note.docx in IEEE conference layout (letter, Times 10pt, two columns,
full-width title + Fig. 1). Equations are native Word OMML. Run: python make_docx.py"""
import re
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt

M = nsdecls("m", "w")
J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT

# ---------- OMML builders ----------
def r(t, roman=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if roman else ""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<m:r>{sty}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr><m:t xml:space="preserve">{t}</m:t></m:r>'
def sub(b, s): return f"<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>"
def frac(n, d): return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"
def bar(e): return f'<m:acc><m:accPr><m:chr m:val="̄"/></m:accPr><m:e>{e}</m:e></m:acc>'
def nary(c, lo, hi, e):
    hide = '<m:supHide m:val="1"/>' if not hi else ""
    return (f'<m:nary><m:naryPr><m:chr m:val="{c}"/>{hide}</m:naryPr><m:sub>{lo}</m:sub>'
            f"<m:sup>{hi}</m:sup><m:e>{e}</m:e></m:nary>")
def omath(x): return parse_xml(f"<m:oMath {M}>{x}</m:oMath>")

lb = bar(r("ℓ"))
ex = lambda a: r("exp", True) + r("(") + a + r(")")
EQ = {
    1: r("s(y∣x)=") + frac(r("1"), r("|y|")) + nary("∑", r("t=1"), r("|y|"),
        r("log", True) + sub(r("p"), r("θ")) + r("(") + sub(r("y"), r("t")) + r("∣x,") + sub(r("y"), r("<t")) + r(")")),
    2: sub(r("c"), r("i")) + r("=") + frac(ex(sub(r("s"), r("i")) + r("/τ")),
        nary("∑", r("j"), "", ex(sub(r("s"), r("j")) + r("/τ")))),
    3: sub(r("M"), r("tok", True)) + r("=2") + r("L") + sub(r("H"), r("kv")) + sub(r("d"), r("h")) + r("b"),
    4: r("S(K)=") + frac(r("K(P+") + lb + r(")"), r("P+K(") + lb + r("−1)")),
    5: sub(r("t"), r("dec", True)) + r("(n)≥") + frac(r("nW"), r("B")),
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
s0.top_margin, s0.bottom_margin = Inches(0.75), Inches(1.0)
COLW = 3.5

def para(align=J, indent=True, after=0, before=0):
    p = doc.add_paragraph(); p.alignment = align
    p.paragraph_format.first_line_indent = Inches(0.14) if indent else None
    p.paragraph_format.space_after = Pt(after); p.paragraph_format.space_before = Pt(before)
    return p

SYM = {r"\mathcal{Y}": "𝒴", r"\bar\ell": "ℓ̄", r"\approx": "≈", r"\times": "×", r"\in": "∈",
       r"\ge": "≥", r"\theta": "θ", r"\tau": "τ"}

def rich(p, text, size=None, bold=False):
    """Text with $math$ (italic letters, _{sub}), **bold** and *italic* spans."""
    for part in re.split(r"(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part: continue
        if part.startswith("$"):
            m = part[1:-1]
            for k, v in SYM.items(): m = m.replace(k, v)
            for tok in re.findall(r"_\{[^}]+\}|_.|[A-Za-z𝒴ℓτθ]̄?|[^A-Za-z_]", m):
                run = p.add_run(tok[2:-1] if tok.startswith("_{") else tok.lstrip("_"))
                if tok.startswith("_"): run.font.subscript = True
                elif tok[0].isalpha() or tok[0] in "𝒴ℓ": run.italic = True
                run.bold = bold
                if size: run.font.size = Pt(size)
        else:
            b, i = part.startswith("**"), part.startswith("*") and not part.startswith("**")
            run = p.add_run(part.strip("*")); run.bold = bold or b; run.italic = i
            if size: run.font.size = Pt(size)
    return p

def body(text, indent=True): return rich(para(indent=indent), text)

def heading(num, title):
    p = para(C, False, after=3, before=6)
    run = p.add_run(f"{num}. {title}"); run.font.small_caps = True

def equation(n):
    p = para(L, False, after=3, before=3)
    ts = p.paragraph_format.tab_stops
    ts.add_tab_stop(Inches(COLW / 2), WD_TAB_ALIGNMENT.CENTER)
    ts.add_tab_stop(Inches(COLW), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t"); p._p.append(omath(EQ[n])); p.add_run(f"\t({n})")

def figure(path, width, caption):
    p = para(C, False, before=4); p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(width))
    rich(para(J, False, after=6), caption, size=8)

def table(num, title, rows, widths, bold_row=None, note=None):
    p = para(C, False, before=6); p.paragraph_format.keep_with_next = True
    p.add_run(f"TABLE {num}").font.size = Pt(8)
    p = para(C, False, after=2); p.paragraph_format.keep_with_next = True
    for run in rich(p, title, size=8).runs: run.font.small_caps = True
    t = doc.add_table(rows=len(rows), cols=len(widths)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    t._tbl.tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
    for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths): gc.set(qn("w:w"), str(int(w * 1440)))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j); cell.width = Inches(widths[j])
            cp = cell.paragraphs[0]; cp.alignment = L if j == 0 else C
            rich(cp, val, size=8, bold=i == bold_row)
    def border(cell, **sides):
        cell._tc.get_or_add_tcPr().append(parse_xml(f"<w:tcBorders {nsdecls('w')}>" + "".join(
            f'<w:{s} w:val="single" w:sz="{sz}" w:color="000000"/>' for s, sz in sides.items()) + "</w:tcBorders>"))
    for j in range(len(widths)):
        border(t.cell(0, j), top=8, bottom=4); border(t.cell(len(rows) - 1, j), bottom=8)
    if note: rich(para(L, False, before=1), note, size=7)
    para(after=2)

# ---------- title block (one column) ----------
p = para(C, False, after=8)
p.add_run("Litmus: KV-Cache Reuse and the Accuracy-Latency Frontier of On-Device Manipulation Detection").font.size = Pt(22)
p = para(C, False); p.add_run("Devansh Gaur").font.size = Pt(11)
p = para(C, False, after=8); p.add_run("University of Waterloo, Waterloo, ON, Canada").italic = True
figure("fig_pipeline.png", 6.6,
       "Fig. 1. Litmus inference path. A two-label gate rejects benign text. Text that passes is scored against "
       "15 candidate labels; the shared prompt prefix is prefilled once into the KV cache (shaded), and each "
       "label is scored by appending its tokens and trimming the cache back to the prefix length $P$.")

# ---------- two-column body ----------
sec = doc.add_section(WD_SECTION.CONTINUOUS)
cols = sec._sectPr.xpath("./w:cols")
c = cols[0] if cols else sec._sectPr.makeelement(qn("w:cols"), {})
c.set(qn("w:num"), "2"); c.set(qn("w:space"), str(int(0.25 * 1440)))
if not cols: sec._sectPr.append(c)

p = para(indent=False, after=4)
a = p.add_run("Abstract: "); a.bold = a.italic = True; a.font.size = Pt(9)
rich(p, "Litmus is an on-device macOS utility that labels a selected passage with one of 14 persuasion "
        "techniques, or none, by scoring candidate labels under a local 4-bit large language model. No text is "
        "generated and no data leaves the device. This paper analyzes three inference effects that set the design: "
        "the key-value (KV) cache footprint under grouped-query attention, the computation saved by reusing the "
        "prefix KV state across labels, and the bandwidth bound on autoregressive decoding. On a 300-example "
        "external benchmark, top-1 accuracy rose from 62.7% (3B) to 82.3% (14B, few-shot), and prefix reuse cut "
        "median latency by a factor of 9.0.", size=9, bold=True)
p = para(indent=False, after=4)
a = p.add_run("Index Terms: "); a.bold = a.italic = True; a.font.size = Pt(9)
r9 = p.add_run("Large language models, KV cache, grouped-query attention, on-device inference, roofline model."); r9.bold = True; r9.font.size = Pt(9)

heading("I", "Introduction")
body("Litmus lets a user select any passage and press one shortcut (⌘B) to see the persuasion technique it "
     "applies. The selected text may be private, so inference runs entirely on the device, and the verdict must "
     "appear within about one second. An instruction-tuned large language model (LLM) is therefore used as a "
     "closed-set scorer rather than a generator. The cost of inference then depends on memory traffic and "
     "repeated computation, not on output length. This paper models both from each model's published "
     "configuration and measures the resulting accuracy-latency frontier for 3B, 8B and 14B models.")

heading("II", "System Overview")
body("A local server runs the model through Apple MLX [1] with 4-bit weights (Fig. 1). A gate first scores "
     "the labels *manipulation* and *none*; text classified as *none* exits immediately. Otherwise the technique "
     "stage scores $K$=15 labels: 14 techniques and *none*. The default model is Llama-3.2-3B-Instruct [2]; "
     "Llama-3.1-8B and Qwen2.5-14B [3] are selectable. All measurements used an Apple M4 Pro with 24 GB of "
     "unified memory.")

heading("III", "Scoring as Constrained Decoding")
body("For input $x$ and each label $y \\in \\mathcal{Y}$, the scorer computes the length-normalized log-likelihood")
equation(1)
body("Normalization by $|y|$ removes the preference for short labels (1 to 7 tokens). A tempered softmax "
     "converts the scores into confidences,", indent=False)
equation(2)
body("with $τ$=0.25. The verdict is the arg max, and every label with $c_{i} \\ge$ 0.15 joins a *mixture* "
     "field.", indent=False)

heading("IV", "KV-Cache Memory")
body("Each attention layer caches a key and a value vector for every processed token. The cache size per token is")
equation(3)
body("with $L$ layers, $H_{kv}$ key-value heads, head dimension $d_{h}$ and $b$ bytes per element. Grouped-query "
     "attention (GQA) [4] shares each key-value head across $H/H_{kv}$ query heads, which divides the cache by "
     "that factor relative to multi-head attention (MHA). Table I lists each checkpoint's configuration.", indent=False)
table("I", "Model Configurations (4-bit Weights, fp16 KV Cache)",
      [("Model", "$L$", "$H$", "$H_{kv}$", "$W$ (GB)", "KiB/token"),
       ("Llama-3.2-3B", "28", "24", "8", "1.81", "112"),
       ("Llama-3.1-8B", "32", "32", "8", "4.52", "128"),
       ("Qwen2.5-14B", "48", "40", "8", "8.31", "192")],
      [1.05, 0.35, 0.35, 0.4, 0.6, 0.7])
body("The 1446-token few-shot prompt occupies 271 MiB on the 14B, against 1.32 GiB under MHA (Fig. 2). Beside "
     "8.31 GB of weights the cache is minor, and the 14B left about 44% of memory free. Capacity did not limit "
     "model choice; latency did.")
figure("fig_kv.png", 3.1, "Fig. 2. KV-cache size from Eq. (3). Dotted: counterfactual 14B with MHA. "
       "Vertical line: deployed prompt length.")

heading("V", "Prefix KV Reuse")
body("All labels share a prefix of $P$ tokens. Under a causal mask the prefix keys and values depend only on "
     "earlier prefix tokens, so they are identical for every label. The deployed scorer prefills the prefix once, "
     "appends label $i$, reads its log-probabilities, and trims the cache back to length $P$. The result is "
     "numerically identical to $K$ independent passes, and forwarded tokens fall by")
equation(4)
body("For $P$=1446 the scorer forwards 1493 tokens instead of 21,752 (14.6×). When measured ($K$=12, "
     "$P \\approx$ 350), Eq. (4) predicted 11.0× and median latency fell from 4014 to 446 ms, a factor of 9.0 "
     "(Fig. 3). The gap reflects fixed per-call overhead and label tokens attending over the full prefix.", indent=False)
figure("fig_reuse.png", 3.1, "Fig. 3. Forward-token reduction from Eq. (4) against the measured median speedup.")

heading("VI", "Prefill Versus Decode")
body("Prefill processes all prompt tokens in parallel, reusing each weight read across many tokens, and is "
     "compute-bound [5]. Decode emits one token per step and must stream every weight per step, so in roofline "
     "terms [6] it is bandwidth-bound:")
equation(5)
body("With $W$=8.31 GB and $B$=273 GB/s the 14B needs at least 30.4 ms per token (3B: 6.6 ms). Log-prob "
     "scoring decodes nothing. DeepSeek-R1-Distill-Qwen-14B [7] must decode a chain of thought first; it averaged "
     "35 s per scan, which bounds its mean chain at about 1150 tokens (Fig. 4).", indent=False)
figure("fig_decode.png", 3.1, "Fig. 4. Decode lower bound from Eq. (5). The 14B scorer sits at $n$=0.")

heading("VII", "Experimental Results")
body("The benchmark holds 300 externally labeled examples from dark-pattern, logical-fallacy and clickbait "
     "corpora, none authored or tuned on. Intervals are Wilson 95% [8]; MCC is reported because classes are "
     "imbalanced (Table II, Fig. 5).")
table("II", "Held-Out Benchmark ($N$=300), All On-Device",
      [("Configuration", "Top-1", "95% CI", "MCC", "Lat. (s)"),
       ("3B, few-shot (shipped)", "62.7", "57.1, 67.9", "0.55", "0.6"),
       ("3B, gate fix, cleaned*", "66.9", "61.3, 72.1", "n/a", "0.6"),
       ("8B, zero-shot", "69.0", "63.6, 74.0", "0.634", "7.4"),
       ("14B, zero-shot", "79.3", "74.4, 83.5", "0.752", "12.7"),
       ("14B, few-shot", "82.3", "77.6, 86.2", "0.787", "7 to 13"),
       ("14B, LoRA (partial)†", "78.9", "n/a", "n/a", "n/a"),
       ("R1-Distill-14B‡", "76.2", "61.5, 86.5", "n/a", "35")],
      [1.2, 0.48, 0.72, 0.45, 0.55], bold_row=5,
      note="*$N$=290 after removing 10 mislabeled items. †123 of 300. ‡Balanced 42-example subset.")
figure("fig_acc.png", 3.1, "Fig. 5. Top-1 accuracy against latency. Hollow marker: balanced 42-example subset.")
body("Accuracy rose monotonically with size (62.7%, 69.0%, 79.3%), so the 3B result is a capacity limit. "
     "Few-shot exemplars [9] added three points on the 14B but tied on the 3B. LoRA [10] on 503 imbalanced "
     "examples did not beat few-shot, and self-consistency voting [11] gained nothing once labels were cleaned. "
     "On the 3B, top-2 accuracy is 73.0% against 64.0% top-1, so most errors name a co-present technique.")

heading("VIII", "Limitations and Conclusion")
body("R1 used a balanced subset, the LoRA run was partial, and the benchmark covers six of 14 techniques. "
     "Likelihood scoring removes decode, prefix reuse removes most repeated prefill, and GQA keeps the cache "
     "small. The 3B serves the 0.6 s shortcut and the 14B an opt-in deep scan at 82.3%.")


# ---------- references ----------
p = para(C, False, after=3, before=6); p.paragraph_format.keep_with_next = True
p.add_run("References").font.small_caps = True
refs = [
    "A. Hannun *et al.*, “MLX: Efficient and flexible machine learning on Apple silicon,” GitHub repository, ml-explore/mlx, 2023.",
    "A. Grattafiori *et al.*, “The Llama 3 herd of models,” arXiv:2407.21783, 2024.",
    "Qwen Team, “Qwen2.5 technical report,” arXiv:2412.15115, 2024.",
    "J. Ainslie *et al.*, “GQA: Training generalized multi-query transformer models from multi-head checkpoints,” in *Proc. EMNLP*, 2023.",
    "R. Pope *et al.*, “Efficiently scaling transformer inference,” in *Proc. MLSys*, 2023.",
    "S. Williams, A. Waterman, and D. Patterson, “Roofline: An insightful visual performance model for multicore architectures,” *Commun. ACM*, vol. 52, no. 4, pp. 65‑76, 2009.",
    "DeepSeek-AI, “DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning,” arXiv:2501.12948, 2025.",
    "E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209‑212, 1927.",
    "T. Brown *et al.*, “Language models are few-shot learners,” in *Proc. NeurIPS*, 2020.",
    "E. J. Hu *et al.*, “LoRA: Low-rank adaptation of large language models,” in *Proc. ICLR*, 2022.",
    "X. Wang *et al.*, “Self-consistency improves chain of thought reasoning in language models,” in *Proc. ICLR*, 2023.",
]
for i, ref in enumerate(refs, 1):
    p = para(J, False)
    p.paragraph_format.left_indent = Inches(0.28); p.paragraph_format.first_line_indent = Inches(-0.28)
    rich(p, f"[{i}]\t{ref}", size=8)

doc.save("litmus_note.docx")
print("saved")
