"""Builds litmus_paper.docx in IEEE conference layout (letter, Times 10pt, two columns,
full-width title + Fig. 1). Equations are native Word OMML. Citations [@key] number in order
of first use. Run: python make_docx.py"""
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
    2: r("S(K)=") + frac(r("K(P+") + lb + r("−1)"), r("P+K(") + lb + r("−1)")),
    3: sub(r("t"), r("pass", True)) + r("(m)≥") + frac(r("W"), r("B")),
    4: sub(r("M"), r("tok", True)) + r("=2") + r("L") + sub(r("H"), r("kv")) + sub(r("d"), r("h")) + r("b"),
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

CITED = []
def cite(m):
    k = m.group(1)
    if k not in CITED: CITED.append(k)
    return f"[{CITED.index(k) + 1}]"

def rich(p, text, size=None, bold=False):
    """Text with $math$ (italic letters, _{sub}), **bold**, *italic* and [@key] citation spans."""
    text = re.sub(r"\[@(\w+)\]", cite, text)
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
p.add_run("Cross-Request Prefix Caching and the Cost of Short Forward Passes "
          "in On-Device LLM Classification").font.size = Pt(20)
p = para(C, False); p.add_run("Devansh Gaur").font.size = Pt(11)
p = para(C, False, after=8); p.add_run("University of Waterloo, Waterloo, ON, Canada").italic = True
figure("fig_hero.png", 6.6,
       "Fig. 1. (a) Median latency of the 15-label technique stage as each source of repeated computation is "
       "removed (300 inputs; 14B label reuse on 30; no reuse on 10 for the 3B and 3 for the 14B). (b) Latency "
       "of one forward pass of $m$ new tokens over a cached 1.4k-token prefix (median of 9). Dashed lines: the "
       "bandwidth bound of Eq. (3). The cost of short passes rises in steps rather than following that bound.")

# ---------- two-column body ----------
sec = doc.add_section(WD_SECTION.CONTINUOUS)
cols = sec._sectPr.xpath("./w:cols")
c = cols[0] if cols else sec._sectPr.makeelement(qn("w:cols"), {})
c.set(qn("w:num"), "2"); c.set(qn("w:space"), str(int(0.25 * 1440)))
if not cols: sec._sectPr.append(c)

p = para(indent=False, after=4)
a = p.add_run("Abstract: "); a.bold = a.italic = True; a.font.size = Pt(9)
rich(p, "A local language model used as a closed-set classifier spends most of its time repeating work. "
        "About 99% of every prompt is identical across inputs, and 15 labels are each scored against it. We remove this redundancy in three steps: reusing the prefix across labels, caching it "
        "across requests, and scoring all labels in a single tree-masked pass. Each step is measured on up to "
        "300 inputs with 3B and 14B models on an Apple M4 Pro. Median end-to-end latency falls from 3.53 s to "
        "0.17 s (3B) and from 15.3 s to 0.68 s (14B); 1% of decisions change, all at ties. The third step "
        "follows from a measurement. After caching, the remaining work is a series of one- to six-token passes "
        "whose cost is set by the 32-row tiling of 4-bit matrix multiplication rather than by memory "
        "bandwidth. Packing the labels into one pass exploits that structure and halves the remaining latency.",
     size=9, bold=True)
p = para(indent=False, after=4)
a = p.add_run("Index Terms: "); a.bold = a.italic = True; a.font.size = Pt(9)
r9 = p.add_run("Large language models, KV cache, prefix caching, quantization, on-device inference.")
r9.bold = True; r9.font.size = Pt(9)

heading("I", "Introduction")
body("An instruction-tuned language model can classify text without generating any. The prompt holds "
     "the task definition, worked examples and the input, and the classifier asks how probable each "
     "candidate label is as the continuation [@zhao]. Our system uses it to name the persuasion "
     "technique in a passage, entirely on a laptop. Its prompt is 1446 tokens "
     "long. The passage being classified typically occupies about 19 of them.")
body("Everything before the passage is the same for every request, and everything before a label is the "
     "same for every label, yet a direct implementation recomputes both. The deployed 14B model needed "
     "15 s per classification, which is why it was never made the default despite being far more accurate "
     "than the 3B. We ask where that time goes and how much of it is necessary.")
body("We measure the effect of caching the input-independent prefix across requests, on top of reuse "
     "across labels. We show that the short passes left afterwards have a step-shaped cost and trace it to "
     "the 4-bit matrix-multiplication kernel. We then use that cost structure to design a packed label pass, "
     "implement it, and test it against its prediction.")

heading("II", "Related Work")
body("Reusing attention state is established practice in LLM serving. PagedAttention [@vllm] shares KV "
     "blocks between sequences, SGLang [@sglang] keeps prefixes in a radix tree, Prompt Cache [@promptcache] "
     "precomputes the attention state of reusable prompt modules, and Hydragen [@hydragen] batches "
     "attention over a shared prefix. These systems target throughput across many users; here one user waits "
     "for one decision, and the shared prefix is known in advance. Tree-shaped attention masks verify many candidate continuations at once in speculative decoding "
     "[@specinfer]; we apply the same idea to a closed label set.")

heading("III", "Classification by Label Scoring")
body("For input $x$ and each label $y \\in \\mathcal{Y}$, the classifier computes the length-normalized "
     "log-likelihood")
equation(1)
body("and returns the arg max, with a tempered softmax ($τ$=0.25) for confidence. A gate first scores "
     "*yes* and *no* for whether the text is manipulative and returns *none* when $p$(*no*) ≥ 0.65. Text "
     "that passes reaches the technique stage, which scores $K$=15 labels of 1 to 7 tokens (62 in total). "
     "Both stages run Llama-3.2-3B-Instruct [@llama] or Qwen2.5-14B-Instruct [@qwen] with 4-bit weights "
     "under MLX [@mlx] on an Apple M4 Pro with 24 GB of unified memory and 273 GB/s of memory bandwidth. "
     "The 3B has 28 layers and $W$=1.81 GB of weights, the 14B 48 layers and 8.31 GB; both use 8 "
     "key-value heads of dimension 128.", indent=False)

heading("IV", "Removing Redundant Work")
body("*Across labels.* Under a causal mask, the keys and values of the $P$ prompt tokens do not depend on "
     "anything that follows them. The deployed scorer prefills the prompt once, appends each label, reads "
     "its log-probabilities and trims the cache back to length $P$. The final label token is never an input, "
     "so $K$ independent passes forward $K(P+\\bar\\ell-1)$ tokens, and reuse reduces this by")
equation(2)
body("where $\\bar\\ell$ is the mean label length. For $P$=1446 and $K$=15, Eq. (2) predicts 14.6×. The "
     "measured ratios were 15.4× (3B, 10 inputs) and 15.8× (14B, 3 inputs). The excess is attention cost, "
     "quadratic in $P$, which Eq. (2) does not count.", indent=False)
body("*Across requests.* Label reuse still prefills the full prompt for every request, and 98.7% of that "
     "prompt (98.5% for the gate) never changes. We prefill the shared part once at startup (2.7 s "
     "on the 3B, 13.8 s on the 14B) and keep its cache resident. Each request forwards only its suffix (the passage and "
     "closing template) and trims back afterwards. The gate, whose labels are "
     "single tokens, reduces to one pass of about 19 tokens. Table I gives the latency, including the packed "
     "pass of Section V.")
table("I", "End-to-End Latency (s) on 300 Inputs",
      [("Stage", "3B base", "cache", "pack", "14B base", "cache", "pack"),
       ("Gate, p50", "1.56", "0.046", "0.044", "6.87", "0.172", "0.171"),
       ("Technique, p50", "1.95", "0.260", "0.127", "8.42", "0.966", "0.512"),
       ("Both, p50", "3.53", "0.306", "0.170", "15.33", "1.140", "0.683"),
       ("Both, p95", "5.80", "0.382", "0.241", "17.94", "1.506", "0.995")],
      [0.86, 0.46, 0.41, 0.41, 0.54, 0.41, 0.41], bold_row=3,
      note="Base: deployed (label reuse). Cache: + request prefix. Pack: + packed labels. 14B base: 30 inputs.")
body("Caching is not bit-exact, since half-precision sums are reordered. Over all "
     "300 inputs on the 3B, three end-to-end decisions (1.0%) changed: two inputs whose gate probability lay "
     "within 0.003 of the threshold, and one exact tie between two techniques. The largest score difference "
     "was 0.031 nats and the largest confidence change 2 points. On the 30 14B inputs with a deployed "
     "reference, no decision changed.")

heading("V", "The Cost of a Short Pass")
body("Once the prefix is cached, the technique stage consists of one 19-token suffix pass followed by 14 "
     "label passes of 1 to 6 tokens. A pass over very few tokens must still read every weight once, which "
     "gives the familiar bandwidth bound [@pope], [@roofline]:")
equation(3)
body("Eq. (3) gives 30.4 ms for the 14B and 6.6 ms for the 3B. Single-token passes measured 38.6 and 10.5 ms, "
     "or 79% and 63% of peak bandwidth. Beyond one token the bound stops describing the data (Fig. 1b). "
     "Cost rises steeply up to 12 tokens, so that an 8-token pass costs 2.8 times a 1-token pass. It then "
     "stays flat to 32 tokens and doubles by 48. Summing the measured cost of each pass in the stage predicts "
     "239 ms (3B) and 878 ms (14B), against 245 and 981 ms measured.", indent=False)
body("To locate the cause, we timed a single matrix multiplication shaped like one 14B MLP weight "
     "(13824×5120) as the number of activation rows $M$ grows (Fig. 2). With 4-bit weights, time is flat for "
     "$M$ ≤ 3, rises roughly linearly to $M$=10, stays flat through $M$=32, and steps up exactly at $M$=33 "
     "and $M$=65. A dense fp16 weight shows none of this structure below 64 rows. The step positions match "
     "the 32-row tile of MLX's quantized matrix-matrix kernel, and the rising segment is consistent with a "
     "per-row matrix-vector kernel at small $M$. Above 32 rows the 4-bit kernel is no faster than fp16.")
figure("fig_qmm.png", 3.0, "Fig. 2. Time of one 13824×5120 matrix multiplication (a 14B MLP weight) "
       "against the number of activation rows $M$ (MLX 0.32.0, median of 15).")
body("The staircase prices the label passes badly: 14 passes, each paying nearly a full weight read, "
     "where a single pass of 47 rows would cost about one tile. A padded batch of labels saved little (0 to 15%), "
     "since its 90 rows each attend over a private copy of the prefix. We therefore place all 47 label tokens in one sequence, ordered by depth, "
     "under a tree-shaped mask that lets each token see the prompt and the earlier tokens of its own label. "
     "Tokens at the same depth share a rotary position, which we apply per depth group. Fig. 1(b) prices "
     "this pass at $t$(48), predicting 0.116 s (3B) and 0.442 s (14B) for the stage. Measured medians over "
     "300 inputs were 0.127 and 0.512 s.")

heading("VI", "Memory")
body("Each token's cache entry occupies")
equation(4)
body("bytes, for $L$ layers, $H_{kv}$ key-value heads of dimension $d_{h}$ and $b$ bytes per element. Both "
     "models use grouped-query attention [@gqa] with $H_{kv}$=8, so the static caches for both stages occupy "
     "0.31 GB (3B) and 0.53 GB (14B), and the packed pass adds only its 47 tokens. Under multi-head attention "
     "($H_{kv}$=40) the 14B caches would occupy 2.7 GB, and the 15-copy batched layout would need 21 GB, "
     "more than the machine holds alongside the weights.", indent=False)

heading("VII", "Consequences")
body("Caching does not remove the trade-off between model sizes: fully optimized, the 3B answers in "
     "0.17 s and the 14B in 0.68 s. It moves the 14B inside an interactive budget, which matters because "
     "the accuracy gap is large. The 300 benchmark examples come "
     "from public dark-pattern [@darkpat], logical-fallacy [@logic] and clickbait [@clickbait] corpora. On the 150 examples that played no "
     "part in tuning, the pipeline reached 86.0% top-1 accuracy with the 14B (95% CI 79.5 to 90.7 [@wilson]) "
     "and 66.0% (58.1 to 73.1) with the 3B. The 14B was right on 34 inputs where the 3B was wrong, and the "
     "reverse held on 4 (exact McNemar test [@mcnemar], $p$ < 10⁻⁶).")
body("A packed 14B classification costs as much as generating about 18 tokens at the measured 37 ms per "
     "token; a distilled reasoning model [@r1] averaged 35 s, roughly 950 tokens, and was not more accurate.")

heading("VIII", "Limitations and Conclusion")
body("Results come from one machine, the slow baselines cover fewer inputs, and accuracy covers five of "
     "14 techniques. Once redundant prefill is gone, latency depends on how many 32-row tiles the remaining tokens occupy, and "
     "packing the labels into one tile halved it again.")

# ---------- references ----------
REFS = {
    "zhao": "Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh, “Calibrate before use: Improving few-shot performance of language models,” in *Proc. ICML*, 2021.",
    "vllm": "W. Kwon *et al.*, “Efficient memory management for large language model serving with PagedAttention,” in *Proc. SOSP*, 2023.",
    "sglang": "L. Zheng *et al.*, “SGLang: Efficient execution of structured language model programs,” in *Proc. NeurIPS*, 2024.",
    "promptcache": "I. Gim *et al.*, “Prompt Cache: Modular attention reuse for low-latency inference,” in *Proc. MLSys*, 2024.",
    "hydragen": "J. Juravsky *et al.*, “Hydragen: High-throughput LLM inference with shared prefixes,” arXiv:2402.05099, 2024.",
    "llama": "A. Grattafiori *et al.*, “The Llama 3 herd of models,” arXiv:2407.21783, 2024.",
    "qwen": "Qwen Team, “Qwen2.5 technical report,” arXiv:2412.15115, 2024.",
    "mlx": "A. Hannun *et al.*, “MLX: Efficient and flexible machine learning on Apple silicon,” GitHub repository, ml-explore/mlx, 2023.",
    "pope": "R. Pope *et al.*, “Efficiently scaling transformer inference,” in *Proc. MLSys*, 2023.",
    "roofline": "S. Williams, A. Waterman, and D. Patterson, “Roofline: An insightful visual performance model for multicore architectures,” *Commun. ACM*, vol. 52, no. 4, pp. 65‑76, 2009.",
    "specinfer": "X. Miao *et al.*, “SpecInfer: Accelerating large language model serving with tree-based speculative inference and verification,” in *Proc. ASPLOS*, 2024.",
    "gqa": "J. Ainslie *et al.*, “GQA: Training generalized multi-query transformer models from multi-head checkpoints,” in *Proc. EMNLP*, 2023.",
    "darkpat": "Y. Yada *et al.*, “Dark patterns in e-commerce: A dataset and its baseline evaluations,” in *Proc. IEEE BigData*, 2022.",
    "logic": "Z. Jin *et al.*, “Logical fallacy detection,” in *Findings of EMNLP*, 2022.",
    "clickbait": "A. Chakraborty, B. Paranjape, S. Kakarla, and N. Ganguly, “Stop clickbait: Detecting and preventing clickbaits in online news media,” in *Proc. ASONAM*, 2016.",
    "mcnemar": "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” *Psychometrika*, vol. 12, no. 2, pp. 153‑157, 1947.",
    "wilson": "E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209‑212, 1927.",
    "r1": "DeepSeek-AI, “DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning,” arXiv:2501.12948, 2025.",
}
assert set(CITED) == set(REFS), set(REFS) ^ set(CITED)
p = para(C, False, after=3, before=6); p.paragraph_format.keep_with_next = True
p.add_run("References").font.small_caps = True
for i, key in enumerate(CITED, 1):
    p = para(J, False)
    p.paragraph_format.left_indent = Inches(0.28); p.paragraph_format.first_line_indent = Inches(-0.28)
    rich(p, f"[{i}]\t{REFS[key]}", size=8)

doc.save("litmus_paper.docx")
print("saved")
