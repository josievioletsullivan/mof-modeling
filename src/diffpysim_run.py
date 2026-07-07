"""Drive the paper's simulator (diffpysim, github.com/yevgenyr/diffpysim) to
generate a *physically* stretched PDF series and extract ground truth.

Unlike a plain x-axis interpolation, diffpysim recomputes each component's PDF
with diffpy.srfit's PDFGenerator at a genuinely expanded lattice (linear strain
0 -> max_strain across `steps`), so peaks shift as a real lattice expansion
would. It then linearly combines the components with (optionally shifting)
weights. This is exactly the simulated dataset from the stretched-NMF paper.

`run_diffpysim(...)` runs it in memory (no file dumping / plotting) and returns
the components, the mixture series, and the ground-truth stretch + weight
trajectories parsed from diffpysim's own step labels.
"""
import os
import ast
import io
import contextlib

import numpy as np
from diffpysim.sim import Sim
from diffpysim.userconfig import UserConfig


def run_diffpysim(cif_bank, compounds, steps=30, qmin=0.5, qmax=30.0,
                  uiso=0.005, rmin=0.0, rmax=30.0, rstep=0.01):
    """Return dict with:
        r            : (n_r,) grid
        names        : [component names]
        components   : (K, steps, n_r) each component's PDF along the strain path
        mixture      : (steps, n_r) the linear-combination series
        A0           : (K, n_r) unstrained (step-0) component PDFs = pure signals
        S_true       : (K, steps) stretch factor per component per step (a_i / a_0)
        C_true       : (steps, K) mixing weights per step
    `compounds` maps name -> dict(cifname, weight, max_weight_shift, max_strain).
    """
    parent = os.path.abspath(os.path.join(cif_bank, os.pardir))
    bank_dirname = os.path.basename(os.path.normpath(cif_bank))

    class _Cfg(UserConfig):
        def __init__(self):
            super().__init__()
            self.PDF, self.XRD, self.PLOT, self.DUMP = True, False, False, False
            self.DUMP_CSV = self.DUMP_TXT = False
            self.parent_path = parent
            self.cif_bank_dirname = bank_dirname
            self.experiment = "snmf_sim"
            self.steps = steps
            self.qmin, self.qmax = qmin, qmax
            self.rmin, self.rmax, self.rstep = rmin, rmax, rstep
            self.uiso = self.uiso_max = uiso        # fixed uiso -> stretch is the only x change
            self.NORM_WEIGHTS, self.SHIFT_WEIGHTS = True, True
            self.compounds = {
                name: dict(cifname=c["cifname"], weight=c["weight"],
                           max_weight_shift=c.get("max_weight_shift", 0.0),
                           lat_attrs=self.lat_attrs, max_strain=c["max_strain"],
                           size=0.0)
                for name, c in compounds.items()
            }

    class _Sim(Sim, _Cfg):        # MRO: Sim.super().__init__() resolves to _Cfg
        pass

    cwd = os.getcwd()
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # silence diffpysim's prints
            sim = _Sim()
            names = list(compounds.keys())
            sim.main()            # fills sim.pdfdict + sim.xvector (no dump/plot)
    finally:
        os.chdir(cwd)             # diffpysim chdir's into parent_path; undo it

    r = np.asarray(sim.xvector, float)
    mixname = sim.mixname

    # per-component PDFs and stretch = current lattice a / step-0 lattice a
    components, S_true = [], []
    for name in names:
        series = sim.pdfdict[name]
        arrs = np.array(list(series.values()))                 # (steps, n_r)
        a = np.array([ast.literal_eval(k)["a"] for k in series.keys()])
        components.append(arrs)
        S_true.append(a / a[0])
    components = np.array(components)                           # (K, steps, n_r)
    S_true = np.array(S_true)                                   # (K, steps)
    A0 = components[:, 0, :]                                    # unstrained pures

    # mixture series + weights parsed from the mix step labels
    mix_series = sim.pdfdict[mixname]
    mixture = np.array(list(mix_series.values()))              # (steps, n_r)
    C_true = np.array([ast.literal_eval(k)["w"] for k in mix_series.keys()])  # (steps, K)

    return dict(r=r, names=names, components=components, mixture=mixture,
                A0=A0, S_true=S_true, C_true=C_true)
