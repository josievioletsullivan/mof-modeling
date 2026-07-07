"""Stage 09: quantify fit / match quality (§7).

For each decomposition we have a set of KNOWN true components (obs) and a set of
blindly RECOVERED components (calc).  This stage scores how well the method
recovered them:

  * per matched pair -- weighted profile residual Rw, reduced chi-square, and
    Pearson correlation r;
  * the full N x N similarity matrix between the two sets;
  * the optimal one-to-one assignment via the Hungarian algorithm
    (scipy.optimize.linear_sum_assignment) rather than scoring all N! orders;
  * a single-number read-out -- the product and mean of the matched
    correlations.  Clean recovery => every matched r ~ 1 => product ~ 1.

Cases: §5 NMF (10 phases, no stretch), §6 SNMF (3 phases, synthetic stretch),
§7 SNMF (3 phases, physical diffpysim stretch).  Run stages 05-07 first.
"""
import sys
import json
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import config
from src import fitmetrics as fm

OUT = config.RESULTS


def rows(a):
    """Coerce an array to (n_components, n_points): components are the short axis."""
    a = np.asarray(a, float)
    return a if a.shape[0] <= a.shape[1] else a.T


def load_cases():
    pdf_labels = json.loads((config.PDFS / "labels.json").read_text())
    short = lambda L: [s.split("_")[0] for s in L]
    cases = []

    # §5 NMF -- true pure PDFs vs recovered NMF components (10)
    cases.append(dict(
        name="s5_nmf", title="§5 NMF (10 phases, no stretch)",
        labels=short(pdf_labels),
        obs=rows(np.load(config.PDFS / "G.npy")),
        calc=rows(np.load(OUT / "nmf_components.npy")),
    ))

    # §6 SNMF -- synthetic stretch
    pick6 = ["Ni", "NaCl", "Si"]
    cases.append(dict(
        name="s6_snmf", title="§6 SNMF (3 phases, synthetic stretch)",
        labels=pick6,
        obs=rows(np.load(config.MIXTURES / "snmf_A_true.npy")),
        calc=rows(np.load(OUT / "snmf_components.npy")),
    ))

    # §7 SNMF -- physical diffpysim stretch
    cases.append(dict(
        name="s7_diffpysim", title="§7 SNMF (3 phases, physical stretch)",
        labels=["Ni", "NaCl", "Si"],
        obs=rows(np.load(config.MIXTURES / "diffpysim_A0.npy")),
        calc=rows(np.load(OUT / "diffpysim_components.npy")),
    ))
    return cases


def run_case(c):
    obs, calc, labels = c["obs"], c["calc"], c["labels"]
    n = len(obs)
    res = fm.match(obs, calc)          # obs rows are truth; assigns calc rows
    # order printed rows by the true-component (obs) index
    order = sorted(range(n), key=lambda k: res["row"][k])
    print(f"\n{'='*66}\n{c['title']}   (N={n})\n{'='*66}")
    print(f"{'true phase':12s} {'->calc#':>7s} {'Pearson r':>10s} {'Rw':>8s} {'chi2_nu':>9s}")
    per = []
    for k in order:
        p = res["pairs"][k]
        lab = labels[p["a_index"]]
        print(f"{lab:12s} {p['b_index']:>7d} {p['pearson']:>10.3f} "
              f"{p['rw']:>8.3f} {p['chi2_reduced']:>9.3g}")
        per.append(dict(phase=lab, calc_index=p["b_index"],
                        pearson=round(p["pearson"], 4),
                        rw=round(p["rw"], 4),
                        chi2_reduced=round(p["chi2_reduced"], 6)))
    print(f"{'-'*66}")
    print(f"read-out: product_r={res['product_r']:.3f}  mean_r={res['mean_r']:.3f}"
          f"  min_r={res['min_r']:.3f}  mean_Rw={res['mean_rw']:.3f}"
          f"  max_Rw={res['max_rw']:.3f}")
    np.save(OUT / f"match_{c['name']}_similarity.npy", res["similarity"])
    return dict(name=c["name"], title=c["title"], n=n, pairs=per,
                product_r=round(res["product_r"], 4),
                mean_r=round(res["mean_r"], 4),
                min_r=round(res["min_r"], 4),
                mean_rw=round(res["mean_rw"], 4),
                max_rw=round(res["max_rw"], 4))


def figure(cases, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipped figure: {e})")
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.6))
    if len(cases) == 1:
        axes = [axes]
    for ax, c, res in zip(axes, cases, results):
        S = np.load(OUT / f"match_{c['name']}_similarity.npy")
        im = ax.imshow(S, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        # mark the Hungarian-chosen cells
        m = fm.match(c["obs"], c["calc"])
        for i, j in zip(m["row"], m["col"]):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="red", lw=1.6))
        ax.set_yticks(range(len(c["labels"])))
        ax.set_yticklabels(c["labels"], fontsize=7)
        ax.set_xlabel("recovered component #")
        ax.set_ylabel("true phase")
        ax.set_title(f"{c['title']}\nprod r={res['product_r']:.2f}  "
                     f"mean r={res['mean_r']:.2f}", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, label="|Pearson r|")
    fig.suptitle("§7 match quality — similarity matrices (red = Hungarian assignment)",
                 fontsize=11)
    fig.tight_layout()
    out = OUT / "match_quality_overview.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


def main():
    cases = load_cases()
    results = [run_case(c) for c in cases]
    (OUT / "match_quality_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nsummary -> {OUT / 'match_quality_summary.json'}")
    figure(cases, results)


if __name__ == "__main__":
    main()
