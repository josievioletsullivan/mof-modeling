"""Stage 08: LIGA ab-initio structure solution (§6).

Validation-first: take a §1 structure whose answer we already know, feed the
solver ONLY the pair distances picked from its (clean, forward-computed) G(r)
plus the known lattice parameters, and check LIGA rebuilds the atom positions.

    python pipeline/08_liga.py [TARGET] [--rmax 7.0] [--maxwalltime 120]

TARGET is a stem in data/pdfs (default Cu_mp-30).  Cubic §1 targets only.
"""
import sys
import json
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from diffpy.structure import loadStructure

import config
from src import liga

OUTDIR = config.DATA / "liga"
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="Cu_mp-30",
                    help="stem in data/pdfs (default Cu_mp-30)")
    ap.add_argument("--rmax", type=float, default=7.0,
                    help="max pair distance to pick from G(r) (A)")
    ap.add_argument("--uiso", type=float, default=0.005,
                    help="ADP for the clean LIGA PDF; small = sharp, resolvable "
                         "shells (the stored .gr uses the broad config.UISO)")
    ap.add_argument("--maxwalltime", type=float, default=120)
    ap.add_argument("--rngseed", type=int, default=42)
    args = ap.parse_args()

    gr_path = config.PDFS / f"{args.target}.gr"
    cif_path = config.STRUCTURES / f"{args.target}.cif"
    for p in (gr_path, cif_path):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    stru = loadStructure(str(cif_path))
    latpar = [stru.lattice.a, stru.lattice.b, stru.lattice.c,
              stru.lattice.alpha, stru.lattice.beta, stru.lattice.gamma]
    natoms = len(stru)
    true_elems, formula, radii = liga.composition(cif_path)
    multi = len(set(true_elems)) > 1
    print(f"target        {args.target}")
    print(f"lattice       a,b,c = {latpar[0]:.4f}, {latpar[1]:.4f}, "
          f"{latpar[2]:.4f}  angles = {latpar[3]:.0f},{latpar[4]:.0f},{latpar[5]:.0f}")
    print(f"natoms (cell) {natoms}   formula {formula}")
    if multi:
        print(f"chemistry     radii {radii}")

    # Clean PDF for distance extraction: the stored .gr is deliberately broad
    # (config.UISO) for the decomposition stages, which merges shells into
    # shoulders. Recompute a sharp PDF so every coordination shell is a
    # resolvable peak -- this is the "cleanest PDF" LIGA needs.
    from src.pdf import pdf_from_structure, save_gr
    g = pdf_from_structure(loadStructure(str(cif_path)), uiso=args.uiso)
    r = config.R
    save_gr(OUTDIR / f"{args.target}_clean.gr", g)
    picked = liga.peak_distances(r, g, rmax=args.rmax)
    true_shells = liga.true_shell_distances(cif_path, rmax=args.rmax)
    print(f"\nclean PDF uiso={args.uiso} (stored .gr uses config.UISO={config.UISO})")
    print(f"picked {len(picked)} pair distances from G(r) up to {args.rmax} A:")
    print("  " + "  ".join(f"{d:.3f}" for d in picked))
    print(f"true shells ({len(true_shells)}) from CIF:")
    print("  " + "  ".join(f"{d:.3f}" for d in true_shells))
    # nearest true shell to each picked peak -> extraction error
    if len(true_shells):
        err = [min(abs(true_shells - d)) for d in picked]
        print(f"peak-pick vs true shell: max |dr| = {max(err):.3f} A, "
              f"mean = {np.mean(err):.3f} A")

    # ---- run LIGA ----
    out_xyz = OUTDIR / f"{args.target}_solved.xyz"
    print(f"\nrunning mpbcliga (seed={args.rngseed}, maxwalltime={args.maxwalltime}s) ...")
    solved, stdout, found = liga.solve(
        picked, latpar, natoms, str(out_xyz),
        formula=formula if multi else None,
        radii=radii if multi else None,
        rngseed=args.rngseed, maxwalltime=args.maxwalltime,
        workdir=str(OUTDIR / f"{args.target}_run"),
    )
    (OUTDIR / f"{args.target}_liga.log").write_text(stdout)
    cost = liga.final_cost(stdout)
    print(f"solution found: {found}   final best-cost: {cost}")
    if solved is None:
        raise SystemExit("no structure written -- see log")
    labels, cart = solved
    print(f"placed {len(cart)} atoms -> {out_xyz}")

    # ---- compare to known answer ----
    cmp = liga.compare(cart, cif_path, latpar)
    print(f"\nrecovery vs known CIF (best origin):")
    print(f"  atoms matched : {cmp['n_solved']}/{cmp['n_true']}")
    print(f"  max atom error: {cmp['max_err_A']:.3f} A")
    print(f"  rms atom error: {cmp['rms_err_A']:.3f} A")
    print(f"  solved frac coords:\n{cmp['solved_frac']}")
    print(f"  true   frac coords:\n{cmp['true_frac']}")

    # chemistry: did LIGA label each recovered site with the right element?
    # For a binary on identical sublattices (e.g. rocksalt) the Na<->Cl choice
    # is degenerate under PDF alone -- no scattering weights -- so we accept a
    # correct assignment OR its global element swap.
    chem_ok = True
    chem_note = ""
    if multi:
        matched_true = [true_elems[k] for k in cmp["match"]]
        correct = sum(labels[i] == matched_true[i] for i in range(len(labels)))
        elemset = sorted(set(true_elems))
        swapped = False
        if len(elemset) == 2:
            sw = {elemset[0]: elemset[1], elemset[1]: elemset[0]}
            swap_correct = sum(sw[labels[i]] == matched_true[i]
                               for i in range(len(labels)))
            swapped = swap_correct == len(labels)
        chem_ok = correct == len(labels) or swapped
        chem_note = ("exact" if correct == len(labels)
                     else "up-to-global-swap (PDF-degenerate)" if swapped
                     else "WRONG")
        print(f"  chemistry     : {correct}/{len(labels)} exact -> {chem_note} "
              f"(solved {labels} vs true {matched_true})")

    # The structure written by LIGA is its best answer even when the internal
    # tolcost gate ('Solution found') is unreachable due to a radius-overlap
    # floor, so PASS is judged on recovered geometry (+ chemistry up to swap).
    tol = 0.30
    ok = (cmp["n_solved"] == cmp["n_true"]
          and cmp["max_err_A"] < tol and chem_ok)
    verdict = "PASS" if ok else "FAIL"
    print(f"\n[{verdict}] geometry within {tol} A"
          + (f" + chemistry ({chem_note})" if multi else "")
          + f": {ok}   (LIGA tolcost-gate found={found}, cost={cost:.2g})")

    summary = dict(target=args.target, latpar=latpar, natoms=natoms,
                   formula=formula, rmax=args.rmax, n_picked=int(len(picked)),
                   solution_found=bool(found), final_cost=cost,
                   max_err_A=cmp["max_err_A"], rms_err_A=cmp["rms_err_A"],
                   n_matched=cmp["n_solved"], chemistry_ok=bool(chem_ok),
                   chemistry=chem_note or "n/a (monatomic)", pass_=bool(ok))
    (OUTDIR / f"{args.target}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {OUTDIR / f'{args.target}_summary.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
