"""Command line: fetch the open data, run the study, the real-data check, write report and dashboard.

Usage:
    python -m biogas_twin verify            # ADM1 vs the BSM2 published steady state
    python -m biogas_twin fetch             # Mendeley farm-scale data (12 MB, CC BY 4.0)
    python -m biogas_twin study --workers 6 # simulate, observe, fit, score (~30-60 min)
    python -m biogas_twin real              # false alarms on the real farm data
    python -m biogas_twin report            # results/, docs/img, docs/data.js
    python -m biogas_twin all
"""
from __future__ import annotations

import argparse
import logging
import os

# small matrices everywhere: one BLAS/OpenMP thread per process avoids oversubscription in the pool
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

log = logging.getLogger("biogas_twin")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="biogas_twin", description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["verify", "fetch", "study", "real", "report", "all"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from . import fetch, study

    if args.command == "verify":
        from .verify import bsm2_check
        err = bsm2_check()
        log.info("largest relative deviation from the BSM2 steady state: %.2e", err)
    if args.command in ("fetch", "all"):
        fetch.mendeley(study.DATA.parent / "raw" / "mendeley")
    if args.command in ("study", "all"):
        study.run_study(args.workers)
    if args.command in ("real", "all"):
        from . import report
        from .realdata import check
        report.write_real(check(study.DATA.parent / "raw" / "mendeley"))
    if args.command in ("report", "all"):
        from . import report
        report.write_all()


if __name__ == "__main__":
    main()
