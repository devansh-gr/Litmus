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
    2: r("S(K)=") + frac(r("K(P+") + lb + r(")"), r("P+K(") + lb + r("−1)")),
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
p.add_run("Ninety-Nine Percent Redundant: Prefix Caching and the Cost of Short Forward Passes "
          "in On-Device LLM Classification").font.size = Pt(20)
p = para(C, False); p.add_run("Devansh Gaur").font.size = Pt(11)
p = para(C, False, after=8); p.add_run("University of Waterloo, Waterloo, ON, Canada").italic = True
figure("fig_hero.png", 6.6,
       "Fig. 1. (a) Latency of the 15-label technique stage as each source of repeated computation is removed "
       "(medians over 10 inputs; no-reuse over 3). The hatched bars are predicted from (b), not measured. "
       "(b) Measured latency of a single forward pass of $m$ new tokens over a cached 1.4k-token prefix "
       "(median of 9). Dashed lines: the bandwidth bound of Eq. (3). Short passes do not sit on that bound; "
       "their cost rises in steps.")

# ---------- two-column body ----------
sec = doc.add_section(WD_SECTION.CONTINUOUS)
cols = sec._sectPr.xpath("./w:cols")
c = cols[0] if cols else sec._sectPr.makeelement(qn("w:cols"), {})
c.set(qn("w:num"), "2"); c.set(qn("w:space"), str(int(0.25 * 1440)))
if not cols: sec._sectPr.append(c)

p = para(indent=False, after=4)
a = p.add_run("Abstract: "); a.bold = a.italic = True; a.font.size = Pt(9)
rich(p, "A local language model used as a closed-set classifier spends most of its time repeating work it "
        "has already done. Here, about 99% of every prompt is identical across inputs, and 15 candidate "
        "labels are each scored against it. We remove the two redundancies "
        "in turn, reusing the prefix across labels and then across requests, and measure the effect on 3B "
        "and 14B models running on an Apple M4 Pro. A 14B classification falls from 14.1 s to 1.1 s with "
        "unchanged decisions. What remains is a series of forward passes of one to six tokens, and their "
        "cost does not follow the bandwidth roofline: an 8-token pass costs 2.8 times a 1-token pass, while "
        "every pass of 12 to 32 tokens costs the same. This step-shaped cost accounts for the residual "
        "latency to within 11% and predicts a further 2.2-fold gain from packing all labels into one pass.",
     size=9, bold=True)
p = para(indent=False, after=4)
a = p.add_run("Index Terms: "); a.bold = a.italic = True; a.font.size = Pt(9)
r9 = p.add_run("Large language models, KV cache, prefix caching, on-device inference, roofline model.")
r9.bold = True; r9.font.size = Pt(9)

heading("I", "Introduction")
body("An instruction-tuned language model can classify text without generating any. The prompt holds "
     "the task definition, worked examples and the input, and the classifier asks how probable each "
     "candidate label is as the continuation [@zhao]. The system examined here uses this method to name the "
     "persuasion technique in a passage of text, entirely on the user's laptop. Its prompt is 1446 tokens "
     "long. The passage being classified typically occupies about 19 of them.")
body("That ratio is the subject of this paper. Everything before the passage is the same for every "
     "request, and everything before a label is the same for every label, yet a direct implementation "
     "recomputes both. The deployed 14B model needed 14 s per classification, which is why it was never "
     "made the default despite being 20 points more accurate than the 3B. We ask where those 14 s go, and "
     "how much of that time is necessary.")
body("We show that caching the input-independent prefix across requests, on top of reuse across labels, "
     "cuts 14B latency 12.4-fold without changing a decision. We then measure the short passes that remain, "
     "find a step-shaped cost the roofline does not predict, and use it to explain why batching the labels "
     "barely helped and what would help more.")

heading("II", "Related Work")
body("Reusing attention state is established practice in LLM serving. PagedAttention [@vllm] shares KV "
     "blocks between sequences, SGLang [@sglang] keeps prefixes in a radix tree, Prompt Cache [@promptcache] "
     "precomputes the attention state of reusable prompt modules, and Hydragen [@hydragen] batches "
     "attention over a shared prefix. These systems target throughput across many users. Here there is "
     "one user on a laptop, the metric is single-request latency, and the shared prefix is known in advance.")

heading("III", "Classification by Label Scoring")
body("For input $x$ and each label $y \\in \\mathcal{Y}$, the classifier computes the length-normalized "
     "log-likelihood")
equation(1)
body("and returns the arg max, with a tempered softmax ($τ$=0.25) for confidence. A gate first scores "
     "*yes* and *no* for the question of whether the text is manipulative; only text that passes reaches "
     "the technique stage, which scores $K$=15 labels of 1 to 7 tokens (62 tokens in total). Both stages "
     "run Llama-3.2-3B-Instruct [@llama] or Qwen2.5-14B-Instruct [@qwen] with 4-bit weights under MLX "
     "[@mlx] on an Apple M4 Pro with 24 GB of unified memory and 273 GB/s of memory bandwidth (Table I).",
     indent=False)
table("I", "Models and Prompt Composition",
      [("Model", "$L$", "$H_{kv}$", "$W$ (GB)", "KiB/tok", "Shared / total tok"),
       ("Llama-3.2-3B", "28", "8", "1.81", "112", "1446 / 1465"),
       ("Qwen2.5-14B", "48", "8", "8.31", "192", "1427 / 1446")],
      [0.95, 0.3, 0.4, 0.55, 0.55, 0.95])

heading("IV", "Removing Redundant Work")
body("*Across labels.* Under a causal mask, the keys and values of the $P$ prompt tokens do not depend on "
     "anything that follows them. The deployed scorer therefore prefills the prompt once, appends each label, "
     "reads its log-probabilities and trims the cache back to length $P$. Relative to $K$ independent "
     "passes, the number of forwarded tokens falls by")
equation(2)
body("where $\\bar\\ell$ is the mean label length. For $P$=1446 and $K$=15, Eq. (2) predicts 14.6×. "
     "On identical inputs, the measured ratios were 14.9× (3B) and 15.8× (14B). They exceed the prediction because Eq. (2) counts tokens, while each repeated prefill also pays "
     "attention cost that grows quadratically with $P$.", indent=False)
body("*Across requests.* Label reuse still prefills the full prompt for every request, and 98.7% of that "
     "prompt (98.5% for the gate) never changes. We prefill the shared part once at startup and keep its cache resident. Each "
     "request then forwards only its own suffix (the passage and the closing chat template), scores the "
     "labels, and trims the cache back to the shared length. Table II gives the result. Under this scheme "
     "the gate, whose labels are single tokens, reduces to one forward pass of about 19 tokens.")
table("II", "Median Latency per Stage (s), 10 Inputs",
      [("Stage", "3B base", "3B cached", "14B base", "14B cached"),
       ("Gate, 2 labels", "1.37", "0.042", "6.39", "0.154"),
       ("Technique, 15 labels", "1.80", "0.245", "7.69", "0.981"),
       ("Both stages", "3.17", "0.287", "14.08", "1.135"),
       ("One-time prefill", "n/a", "3.06", "n/a", "12.18")],
      [1.06, 0.58, 0.62, 0.62, 0.62], bold_row=3,
      note="Base: label reuse only. Cached: plus the cross-request prefix cache.")
body("The arg max agreed on all 10 inputs for both models and both stages. The largest score difference "
     "was 0.063 nats, which we attribute to half-precision accumulation order. Every decision was unchanged.")

heading("V", "The Cost of a Short Pass")
body("Once both redundancies are removed, the technique stage consists of one 19-token suffix pass "
     "followed by 14 label passes of 1 to 6 tokens. A pass over very few tokens must still read every "
     "weight once, which gives the familiar bandwidth bound [@pope], [@roofline]:")
equation(3)
body("Eq. (3) gives 30.4 ms for the 14B and 6.6 ms for the 3B. Single-token passes measured 38.6 and "
     "10.5 ms, which correspond to 79% and 63% of peak bandwidth. Beyond one token the roofline stops "
     "describing the data (Fig. 1b). Cost rises steeply between 1 and 12 tokens: an 8-token pass costs "
     "2.8 times a 1-token pass. It then stays flat up to 32 tokens and doubles at 48. This pattern is "
     "consistent with a matrix-vector kernel at small $m$ handing over to MLX's tiled matrix-matrix "
     "kernel, whose row tile is 32.", indent=False)
body("Summing the measured cost of each pass in the stage (interpolating $m$=5) predicts 239 ms for the 3B "
     "and 878 ms for the 14B, against 245 and 981 ms measured. The label passes therefore account for "
     "almost all of the residual time.")
body("We expected that running all 15 labels as a single padded batch would help. It saved nothing on the "
     "3B and 15% on the 14B. Padding turns 47 label tokens into 90 query rows, each attending over its own "
     "copy of the prefix. The staircase points to a better layout: all 47 label tokens packed into one "
     "sequence under a tree-shaped attention mask, as in tree-based speculative decoding [@specinfer]. That "
     "pass costs about $t$(48), giving a predicted 0.12 s (3B) and 0.44 s (14B). It is not yet implemented.")

heading("VI", "Memory")
body("Each token's cache entry occupies")
equation(4)
body("bytes, for $L$ layers, $H_{kv}$ key-value heads of dimension $d_{h}$ and $b$ bytes per element. Both models use grouped-query attention [@gqa] with "
     "$H_{kv}$=8, so the static caches for both stages occupy only 0.31 GB (3B) and 0.53 GB (14B). The "
     "batched experiment holds 15 copies of the prefix, 4.2 GB on the 14B; under multi-head attention "
     "($H_{kv}$=40) the copies alone would need 21 GB and the total would exceed the machine (Fig. 2). "
     "The packed pass needs a single copy.", indent=False)
figure("fig_mem.png", 3.0, "Fig. 2. Unified-memory budget. *MHA is a counterfactual 14B with "
       "$H_{kv}$=40; its weights are left unchanged for comparison.")

heading("VII", "Consequences")
body("With both caches, the 14B classifies in 1.14 s, compared with 3.2 s for the uncached 3B. On 300 "
     "externally labeled examples from public dark-pattern [@darkpat], logical-fallacy [@logic] and "
     "clickbait [@clickbait] corpora, the 14B with few-shot "
     "examples reached 82.3% top-1 accuracy (95% CI 77.6 to 86.2 [@wilson]), against 62.7% "
     "for the 3B. Half of these examples were used to tune decision thresholds, so the numbers compare "
     "configurations rather than estimate field accuracy. The accuracy case for the larger model was already "
     "established. The latency case against it no longer holds.")
body("Generation is the expensive alternative. The 14B decodes at 37 ms per token, so a whole cached "
     "classification costs about as much as 31 generated tokens. A distilled reasoning model [@r1] that "
     "reasons before labeling averaged 35 s, roughly 950 tokens, and was not more accurate.")

heading("VIII", "Limitations and Conclusion")
body("Latencies are medians over 10 inputs on one machine; the first call after a workload switch "
     "occasionally took up to 2.2 s. The packed pass is a prediction, and the benchmark covers five of the "
     "14 techniques. Nearly all of a local LLM classifier's time goes to recomputing one prompt; once that "
     "is removed, the remainder is set by how the hardware treats very short passes.")

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
