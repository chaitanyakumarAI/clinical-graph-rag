"""
evaluation/make_report.py
Build evaluation/RAGAS_Evaluation_Report.pdf from the cached results of run_ragas.py.

Optional: evaluation/results/findings.html — hand-written analysis, inserted
into the report as the "Findings" section.
"""

from __future__ import annotations
import html
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
sys.path.insert(0, EVAL_DIR)
from run_ragas import CONFIGS, METRIC_JUDGES, METRIC_NAMES   # noqa: E402

OUTPUT_PDF = os.path.join(EVAL_DIR, "RAGAS_Evaluation_Report.pdf")

# Validated categorical slots 1–2 (blue, orange); text stays in neutral ink.
SERIES_COLOR = {"baseline": "#2a78d6", "rewrite": "#eb6834"}
SERIES_LABEL = {"baseline": "Without query rewriting", "rewrite": "With query rewriting"}
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

METRIC_LABEL = {
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Answer relevancy",
    "context_recall":     "Context recall",
    "answer_correctness": "Answer correctness",
}
METRIC_MEANING = {
    "faithfulness":       "Are the claims in the answer supported by the retrieved context? (low = hallucination)",
    "answer_relevancy":   "Does the answer address the question that was asked?",
    "context_recall":     "Did retrieval find the passages needed to state the reference answer?",
    "answer_correctness": "How closely does the answer match the reference answer (facts + meaning)?",
}


def _load(name):
    with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _mean(values):
    values = [v for v in values if v == v]   # drop NaN
    return statistics.mean(values) if values else float("nan")


# ── Charts ────────────────────────────────────────────────────────────────────

def _style_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.08)


def _grouped_bars(ax, means_by_config, title=None, value_labels=True):
    n = len(METRIC_NAMES)
    width = 0.36
    for i, config in enumerate(CONFIGS):
        xs = [m + (i - 0.5) * (width + 0.02) for m in range(n)]
        values = [means_by_config[config][metric] for metric in METRIC_NAMES]
        bars = ax.bar(xs, values, width, color=SERIES_COLOR[config],
                      label=SERIES_LABEL[config], edgecolor="white", linewidth=1.5)
        for bar, value in zip(bars if value_labels else [], values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(n))
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRIC_NAMES], color=INK, fontsize=9)
    _style_axes(ax)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)


def _answerable(testset):
    """Questions whose answer is in the documents. RAGAS scores a correct refusal
    ("insufficient information") as 0 relevancy / 0 recall, so the averages leave
    those questions out and the report discusses them separately."""
    return [q for q in testset if q["category"] != "not-in-documents"]


def make_charts(scores, testset):
    ids = [q["id"] for q in _answerable(testset)]
    overall = {c: {m: _mean([scores[c][i][m] for i in ids if i in scores[c]]) for m in METRIC_NAMES}
               for c in CONFIGS}

    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
    _grouped_bars(ax, overall)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.13), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "chart_overall.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=200, sharey=True)
    for ax, style in zip(axes, ("clean", "messy")):
        ids = [q["id"] for q in _answerable(testset) if q["style"] == style]
        means = {c: {m: _mean([scores[c][i][m] for i in ids if i in scores[c]])
                     for m in METRIC_NAMES} for c in CONFIGS}
        _grouped_bars(ax, means, title=f"{style.capitalize()} questions (n={len(ids)})",
                      value_labels=False)   # exact values are in the table below
        ax.set_xticklabels(["Faithful.", "Relevancy", "Recall", "Correct."], fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", ncol=2,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(RESULTS_DIR, "chart_by_style.png"))
    plt.close(fig)
    return overall


# ── HTML ──────────────────────────────────────────────────────────────────────

CSS = """
body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.38; color: #1f1f1e; }
h1 { font-size: 19pt; margin: 0 0 2pt 0; }
h2 { font-size: 13pt; margin: 14pt 0 4pt 0; border-bottom: 1px solid #d6d5cc; padding-bottom: 2pt; }
h3 { font-size: 10.5pt; margin: 9pt 0 3pt 0; }
p.sub { color: #6b6a63; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0; }
th { text-align: left; font-size: 8.5pt; padding: 3pt 4pt; border-bottom: 1.5px solid #9a998f; }
td { font-size: 8.5pt; padding: 3pt 4pt; border-bottom: 1px solid #e4e3dc; vertical-align: top; }
td.num, th.num { text-align: right; white-space: nowrap; }
.newpage { page-break-before: always; }
.kpi { font-size: 20pt; font-weight: bold; }
.muted { color: #6b6a63; }
.small { font-size: 8pt; }
"""


def _markdown_to_html(text: str) -> str:
    """Render the little markdown the LLM uses in answers (**bold**, "- " bullets)."""
    import re
    lines = []
    for line in html.escape(text.strip()).splitlines():
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line.strip())
        if line.startswith(("- ", "• ")):
            line = "• " + line[2:]
        if line:
            lines.append(line)
    return "<br/>".join(lines)


def _fmt_delta(before, after):
    delta = after - before
    arrow = "▲" if delta > 0.005 else "▼" if delta < -0.005 else "•"
    return f"{arrow} {delta:+.2f}"


def build_html(overall, scores, answers, testset, usage, findings_html):
    e = html.escape
    by_id = {q["id"]: q for q in testset}

    kpi_cells = "".join(
        f"<td><div class='muted small'>{METRIC_LABEL[m]}</div>"
        f"<div class='kpi'>{overall['rewrite'][m]:.2f}</div>"
        f"<div class='small'>{_fmt_delta(overall['baseline'][m], overall['rewrite'][m])} vs. "
        f"{overall['baseline'][m]:.2f} without rewriting</div></td>"
        for m in METRIC_NAMES)

    metric_rows = "".join(
        f"<tr><td><b>{METRIC_LABEL[m]}</b></td><td>{e(METRIC_MEANING[m])}</td>"
        f"<td>{e(METRIC_JUDGES[m])}</td></tr>" for m in METRIC_NAMES)

    overall_rows = "".join(
        f"<tr><td>{METRIC_LABEL[m]}</td><td class='num'>{overall['baseline'][m]:.3f}</td>"
        f"<td class='num'>{overall['rewrite'][m]:.3f}</td>"
        f"<td class='num'>{_fmt_delta(overall['baseline'][m], overall['rewrite'][m])}</td></tr>"
        for m in METRIC_NAMES)

    question_rows = ""
    for qid in [q["id"] for q in testset]:
        if qid not in scores["baseline"] or qid not in scores["rewrite"]:
            continue
        q = by_id[qid]
        cells = "".join(
            f"<td class='num'>{scores['baseline'][qid][m]:.2f}&#8202;→&#8202;{scores['rewrite'][qid][m]:.2f}</td>"
            for m in METRIC_NAMES)
        question_rows += (f"<tr><td><b>{qid}</b></td><td>{e(q['question'])}"
                          f"<div class='muted small'>{q['category']} · {q['style']}</div></td>{cells}</tr>")

    rewrite_rows = ""
    for qid in [q["id"] for q in testset]:
        ans = answers["rewrite"].get(qid)
        if not ans:
            continue
        subs = "".join(f"<div class='small muted'>↳ {e(s)}</div>" for s in ans["sub_queries"])
        rewrite_rows += (f"<tr><td><b>{qid}</b></td><td>{e(ans['question'])}</td>"
                         f"<td>{e(ans['rewritten_query'])}{subs}</td></tr>")

    answer_blocks = ""
    for qid in [q["id"] for q in testset]:
        q = by_id[qid]
        answer_blocks += f"<h3>{qid}. {e(q['question'])}</h3>"
        answer_blocks += f"<p class='small'><b>Reference:</b> {e(q['reference'])}</p>"
        for config in CONFIGS:
            ans = answers[config].get(qid)
            if not ans:
                continue
            text = _markdown_to_html(ans["answer"])
            score = scores[config].get(qid, {})
            score_txt = " · ".join(f"{METRIC_LABEL[m]} {score[m]:.2f}" for m in METRIC_NAMES if m in score)
            answer_blocks += (f"<p class='small'><b>{SERIES_LABEL[config]}</b> "
                              f"<span class='muted'>({score_txt})</span><br/>{text}</p>")

    usage_rows = "".join(
        f"<tr><td>{e(step)}</td><td>{e(model)}</td><td class='num'>{u['calls']}</td>"
        f"<td class='num'>{u['tokens']:,}</td></tr>"
        for step, models in usage.items() for model, u in models.items())

    n_clean = sum(q["style"] == "clean" for q in _answerable(testset))
    return f"""
<h1>RAGAS Evaluation: Medical Graph RAG</h1>
<p class="sub">Effect of query rewriting on retrieval and answer quality · {len(testset)} questions × 2 configurations</p>

<table><tr>{kpi_cells}</tr></table>
<p class="small muted">Large numbers: mean score with query rewriting over the
{len(_answerable(testset))} answerable questions (0–1, higher is better). Arrows: change versus the
same pipeline with rewriting switched off.</p>

<h2>Findings</h2>
{findings_html or "<p class='muted'>(No findings file found.)</p>"}

<h2>Overall scores <span class="muted small">({len(_answerable(testset))} answerable questions)</span></h2>
<img src="chart_overall.png" width="500"/>
<table>
<tr><th>Metric</th><th class="num">Without rewriting</th><th class="num">With rewriting</th><th class="num">Change</th></tr>
{overall_rows}
</table>

<h2 class="newpage">Clean vs. messy questions</h2>
<p>{n_clean} answerable questions are written cleanly; {len(_answerable(testset)) - n_clean} are written the way users
often type (typos, abbreviations such as <i>T2DM</i>, <i>BP</i>, <i>AF</i>, <i>pt</i>).
Rewriting is expected to matter most for the messy ones.</p>
<img src="chart_by_style.png" width="500"/>

<h2>Per-question scores (without → with rewriting)</h2>
<table>
<tr><th>ID</th><th>Question</th><th class="num">Faithful.</th><th class="num">Relevancy</th>
<th class="num">Recall</th><th class="num">Correct.</th></tr>
{question_rows}
</table>

<h2>What the rewriter produced</h2>
<table><tr><th>ID</th><th>User question</th><th>Rewritten query (↳ sub-queries)</th></tr>
{rewrite_rows}</table>

<h2>Setup</h2>
<h3>System under test</h3>
<p>Hybrid Graph + Vector RAG (Neo4j knowledge graph + Qdrant vector search, BGE embeddings,
BGE cross-encoder reranker). Answering model: <b>openai/gpt-oss-120b</b> on Groq.
Both configurations are identical except for the <i>use_query_rewrite</i> flag.</p>
<h3>Documents</h3>
<p>Three synthetic educational PDFs (data/pdfs/): a Type 2 Diabetes drug guide, a hypertension
and atrial fibrillation review, and a drug-interaction reference whose facts link the other two.
Other previously ingested documents remained in the database, as in real use.</p>
<h3>Questions</h3>
<p>{len(testset)} questions with hand-written reference answers (evaluation/testset.json):
definitions, single-hop facts, multi-hop facts spanning two documents, and one question whose
answer is not in the documents.</p>
<h3>Metrics</h3>
<table><tr><th>Metric</th><th>What it measures</th><th>Judge model</th></tr>{metric_rows}</table>
<p class="small muted">RAGAS 0.4.3. Judges run on Groq's free tier, where each model allows
200K tokens/day, so metrics are split across two judge models; each metric uses the same judge
for both configurations. Answer relevancy uses local BGE embeddings; answer correctness combines
an LLM fact comparison with embedding similarity.</p>

<h2>Limitations</h2>
<ul>
<li><b>Small sample.</b> {len(testset)} questions; a difference of a few hundredths on one metric
is within run-to-run noise. Treat per-question results as diagnostics, not statistics.</li>
<li><b>LLM judges are noisy</b> and smaller than the answering model. Scores are best used to
compare the two configurations, not as absolute quality grades.</li>
<li><b>Synthetic documents.</b> Real clinical PDFs (tables, scanned pages, long guidelines) are
harder; results here are an upper bound.</li>
<li><b>Single run.</b> LLM outputs vary between runs; repeat the evaluation before relying on
small differences.</li>
</ul>

<h2>LLM usage</h2>
<table><tr><th>Step</th><th>Model</th><th class="num">Calls</th><th class="num">Tokens</th></tr>{usage_rows}</table>

<h2>Appendix: all answers</h2>
{answer_blocks}
"""


def write_pdf(body_html):
    story = pymupdf.Story(html=body_html, user_css=CSS, archive=pymupdf.Archive(RESULTS_DIR))
    writer = pymupdf.DocumentWriter(OUTPUT_PDF)
    mediabox = pymupdf.paper_rect("a4")
    content = mediabox + (48, 48, -48, -54)
    page_no, more = 0, True
    while more:
        page_no += 1
        device = writer.begin_page(mediabox)
        more, _ = story.place(content)
        story.draw(device)
        writer.end_page()
    writer.close()

    # Page numbers in the footer, then shrink the file: the layout engine embeds
    # complete fonts on every page, so subset them and drop duplicate objects.
    doc = pymupdf.open(OUTPUT_PDF)
    for i, page in enumerate(doc, 1):
        page.insert_text((mediabox.width / 2 - 20, mediabox.height - 28),
                         f"Page {i} of {len(doc)}", fontsize=8, color=(0.42, 0.42, 0.39))
    doc.subset_fonts()
    doc.ez_save(OUTPUT_PDF + ".tmp", garbage=4)
    doc.close()
    os.replace(OUTPUT_PDF + ".tmp", OUTPUT_PDF)


def main():
    testset = json.load(open(os.path.join(EVAL_DIR, "testset.json"), encoding="utf-8"))
    scores  = {c: _load(f"scores_{c}.json")  for c in CONFIGS}
    answers = {c: _load(f"answers_{c}.json") for c in CONFIGS}
    usage_path = os.path.join(RESULTS_DIR, "usage.json")
    usage = json.load(open(usage_path)) if os.path.exists(usage_path) else {}
    findings_path = os.path.join(RESULTS_DIR, "findings.html")
    findings = open(findings_path, encoding="utf-8").read() if os.path.exists(findings_path) else ""

    overall = make_charts(scores, testset)
    write_pdf(build_html(overall, scores, answers, testset, usage, findings))
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
