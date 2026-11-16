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
       "removed (300 inputs, no-reuse baseline on 10 for the 3B and 3 for the 14B). (b) Latency "
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
rich(p, "A local language model used as a closed-set classifier spends most of its time repeating work: "
        "about 99% of each prompt is identical across inputs, and each of 15 labels is scored against it. "
        "We remove this redundancy in three steps. First, the prompt is computed once and shared by all labels. "
        "Second, its fixed part is cached across requests. Third, all labels are scored together in one "
        "tree-masked pass. On an Apple M4 Pro with 3B and 14B models and 300 inputs, the second and third steps "
        "reduce warm median latency of the full classifier, relative to the first step alone, from 3.53 s to "
        "0.17 s (3B) and from 15.2 s to 0.68 s (14B). No 14B decision changes, while three of 300 3B decisions do, "
        "each at a near-tie. Packing helps because, after caching, the remaining work is a series of passes of "
        "one to six tokens whose cost rises in steps, consistent with the 32-row tiles of MLX's 4-bit matrix "
        "multiplication. One packed pass therefore halves the remaining technique-stage latency.",
     size=9, bold=True)
p = para(indent=False, after=4)
a = p.add_run("Index Terms: "); a.bold = a.italic = True; a.font.size = Pt(9)
r9 = p.add_run("Large language models, KV cache, prefix caching, quantization, on-device inference.")
r9.bold = True; r9.font.size = Pt(9)

heading("I", "Introduction")
body("A 14-billion-parameter language model running on a laptop took 15 s to classify a passage of about "
     "19 tokens. It generated no text. It only chose among 15 fixed labels, by asking how probable each label "
     "is as the continuation of a prompt holding the task definition, worked examples and the passage [@zhao]. "
     "That prompt is 1446 tokens long, and all of it except the passage is identical for every request. Most "
     "of the 15 s was spent recomputing it.")
body("Removing that repetition exposed a second, less obvious cost. Each label is only a few tokens, yet a "
     "pass over 8 tokens cost 2.8 times a pass over one, while a 32-token pass cost no more than a 12-token "
     "one. This step-shaped cost is consistent with the tiling of the 4-bit matrix-multiplication kernel, and "
     "it makes many short passes far more expensive than one longer pass. We exploit it by scoring all labels "
     "in one tree-masked pass. Warm median latency fell from 15.2 s to 0.68 s for the 14B model and from "
     "3.53 s to 0.17 s for the 3B, which made the far more accurate 14B usable interactively.")

heading("II", "Related Work")
body("Reusing attention state is established practice in LLM serving. PagedAttention [@vllm] and SGLang "
     "[@sglang] share cached prefixes across requests, and Hydragen [@hydragen] batches attention over a "
     "shared prefix. Prompt Cache [@promptcache], like our cross-request cache, precomputes prompt modules "
     "fixed in advance to cut time to first token. We differ in what "
     "remains after reuse: one request scoring short fixed labels on Apple silicon, where per-pass cost dominates. Tree-shaped attention masks verify many candidate "
     "continuations at once in speculative decoding [@specinfer], and the small-$M$ cost of MLX 4-bit matrix "
     "multiplication was reported in that setting [@mlxissue]. We apply both observations to a closed label set.")

heading("III", "Classification by Label Scoring")
body("For a passage such as *Only 2 left, order now*, the classifier scores candidate labels such as *fomo* "
     "and *false-urgency* as continuations using length-normalized log-likelihood. For input $x$ and each label $y \\in \\mathcal{Y}$ the score is")
equation(1)
body("The classifier selects the highest-scoring label and converts scores to confidences with a tempered softmax ($τ$=0.25). A gate first scores "
     "*yes* and *no* for whether the text is manipulative and returns *none* when $p$(*no*) ≥ 0.65. Text "
     "that passes reaches the technique stage, which scores $K$=15 labels of 1 to 7 tokens (62 in total). "
     "Both stages run Llama-3.2-3B-Instruct [@llama] or Qwen2.5-14B-Instruct [@qwen] with 4-bit weights "
     "under MLX [@mlx] on an Apple M4 Pro with 24 GB of unified memory and 273 GB/s of memory bandwidth. "
     "The 3B has 28 layers and $W$=1.81 GB of weights, the 14B 48 layers and 8.31 GB. Both use 8 "
     "key-value heads of dimension 128.", indent=False)

heading("IV", "Removing Redundant Work")
body("*Across labels.* Scoring 15 labels independently runs the 1446-token prompt through the model "
     "15 times. Under a causal mask, the keys and values of the $P$ prompt tokens do not depend on "
     "anything that follows them. The deployed scorer prefills the prompt once, appends each label, reads "
     "its log-probabilities and trims the cache back to length $P$. The final label token is never an input, "
     "so $K$ independent passes forward $K(P+\\bar\\ell-1)$ tokens, and reuse reduces this by")
equation(2)
body("where $\\bar\\ell$ is the mean label length. For $P$=1446 and $K$=15, Eq. (2) predicts 14.6×. The "
     "measured ratios were 15.4× (3B, 10 inputs) and 15.8× (14B, 3 inputs). The excess is attention cost, "
     "which Eq. (2) omits.", indent=False)
body("*Across requests.* Label reuse still prefills the full prompt for every request, and 98.7% of that "
     "prompt (98.5% for the gate) never changes. We prefill the shared part once at startup (2.7 s "
     "on the 3B, 13.8 s on the 14B) and keep its cache resident. With grouped-query attention [@gqa] the caches for both stages occupy only 0.31 GB (3B) and 0.53 GB (14B). Each request forwards only its suffix (the passage and "
     "closing template) and trims back afterwards (Table I). The gate, whose labels are "
     "single tokens, reduces to one pass of about 19 tokens.")
table("I", "Full-Path Warm Latency (s), 300 Inputs",
      [("Stage", "3B base", "cache", "pack", "14B base", "cache", "pack"),
       ("Gate, p50", "1.56", "0.046", "0.044", "6.73", "0.172", "0.171"),
       ("Technique, p50", "1.95", "0.260", "0.127", "8.49", "0.966", "0.512"),
       ("Both, p50", "3.53", "0.306", "0.170", "15.21", "1.140", "0.683"),
       ("Both, p95", "5.80", "0.382", "0.241", "17.53", "1.506", "0.995")],
      [0.86, 0.46, 0.41, 0.41, 0.54, 0.41, 0.41], bold_row=3,
      note="Base: deployed (label reuse). Cache: + request prefix. Pack: + packed labels. Full path: gate and technique stage on every input. In use, gate-rejected passages skip the technique stage.")
body("Caching reorders half-precision sums, so it is not bit-exact. Over "
     "300 inputs on the 3B, three final decisions (1.0%) changed, all through caching and none through packing: two inputs whose gate probability lay "
     "within 0.003 of the threshold, and one exact tie between two techniques. On the real labels, the largest score difference "
     "between the base and the final path was 0.031 nats. On the 14B, no decision "
     "changed in 300. Median paired speedups over the base were 20.4× (3B) and 22.0× (14B).")

heading("V", "The Cost of a Short Pass")
body("Once the prefix is cached, the technique stage consists of one 19-token suffix pass followed by 14 "
     "label passes of 1 to 6 tokens. A pass over very few tokens must still read every weight once, which "
     "gives the familiar bandwidth bound [@pope], [@roofline]:")
equation(3)
body("Eq. (3) gives 30.4 ms for the 14B and 6.6 ms for the 3B. Single-token passes measured 38.6 and 10.5 ms, "
     "or 79% and 63% of peak bandwidth. Beyond one token the bound stops describing the data (Fig. 1b). "
     "Cost rises steeply to 12 tokens, stays flat to 32 and "
     "doubles by 48. Summing the measured cost of each pass in the stage predicts "
     "239 ms (3B) and 878 ms (14B), against 260 and 966 ms measured.", indent=False)
body("The plateau is the useful part. If a 32-token pass costs the same as a 12-token one, up to 20 "
     "extra tokens ride along almost free, and 14 separate label passes waste that capacity. To locate the "
     "cause, we timed a single matrix multiplication shaped like one 14B MLP weight "
     "(13824×5120) as the number of activation rows $M$ grows (Fig. 2). With 4-bit weights, time rises roughly "
     "linearly to $M$=10, then steps up exactly at $M$=33 and $M$=65. A dense fp16 weight shows none of this structure below 64 rows. The step positions match "
     "the 32-row tile of MLX's quantized matrix-matrix kernel, and the rising segment is consistent with a "
     "per-row matrix-vector kernel at small $M$.")
figure("fig_qmm.png", 3.0, "Fig. 2. Time of one 13824×5120 matrix multiplication (a 14B MLP weight) "
       "against the number of activation rows $M$ (MLX 0.32.0, median of 15). A dependent chain of up and down "
       "projections, synchronized once per 40 operations, agrees within 7%.")
body("The staircase prices the label passes badly: 14 passes, each paying nearly a full weight read, "
     "where a single pass of 47 rows spans two tiles. We therefore place all 47 label tokens in one sequence, ordered by depth, "
     "under a tree-shaped mask that lets each token see the prompt and the earlier tokens of its own label but no other label's. "
     "Tokens at the same depth share a rotary position. Fig. 1(b) prices "
     "this pass at $t$(48), predicting 0.116 s (3B) and 0.442 s (14B) for the stage. Measured medians over "
     "300 inputs were 0.127 and 0.512 s.")
body("With synthetic label sets of 8 to 96 tokens (5-token labels, 50 benchmark inputs), packed time was "
     "flat from 16 to 32 and from 48 to 64 tokens, and the median gain over sequential scoring grew from "
     "1.0× at 8 tokens to 3.5× (3B) and 2.8× (14B) at 96. On these synthetic labels, packed and sequential scores differed by at most "
     "0.067 nats, whereas a causal mask across labels or consecutive positions changed them by 0.6 to 13 nats.")

heading("VI", "Consequences")
body("Caching does not remove the trade-off between model sizes: the 3B answers in "
     "0.17 s and the 14B in 0.68 s (resident warm cache, p99 1.30 s). It moves the 14B inside an interactive budget, which matters given "
     "the accuracy gap. The 300 benchmark examples come "
     "from public dark-pattern [@darkpat], logical-fallacy [@logic] and clickbait [@clickbait] corpora. On the 150 examples that played no "
     "part in tuning, the pipeline reached 86.0% top-1 accuracy with the 14B (95% CI 79.5 to 90.7 [@wilson]) "
     "and 66.0% (58.1 to 73.1) with the 3B, and the 14B won 34 of 38 "
     "discordant inputs ($p$ < 10⁻⁶, exact McNemar test [@mcnemar]).")

heading("VII", "Limitations and Conclusion")
body("For this short-input workload, once redundant prefix work is removed, the cost of short forward "
     "passes sets latency, and packing the labels into one tree-masked pass halved the remaining technique-stage latency. Results come from one "
     "machine, no-reuse baselines cover 10 and 3 inputs, and accuracy covers five of 14 techniques. Code and data: github.com/devansh-gr/Litmus (tag paper-v12).")

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
    "mlxissue": "A. V. Propriis, “quantized_matmul does not amortize the weight read for small M, while the bf16 path does,” ml-explore/mlx GitHub issue #4265, Aug. 2026.",
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
