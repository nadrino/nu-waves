#!/usr/bin/env python3
"""Load OA2026 ROOT minituples into a nu-waves batch without GUNDAM.

Only the oscillation inputs are reproduced, not GUNDAM sample selections,
weights, friend trees or probability binning. See README.md in this directory.
"""

import argparse
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np


def configureNuWaves(rawPath):
    """Accept a checkout, its src directory, or the package directory itself."""
    candidate = Path(rawPath).expanduser().resolve()
    if candidate.name == "nu_waves":
        candidate = candidate.parent
    if (candidate / "src" / "nu_waves").is_dir():
        candidate = candidate / "src"
    if not (candidate / "nu_waves" / "__init__.py").is_file():
        raise ValueError(f"Cannot find nu_waves in {rawPath}")
    sys.path.insert(0, str(candidate))


def resolveInputs(args):
    """Read just model.tree and model.filePathList from an OA dataset list."""
    sources = [(value, args.tree) for value in args.inputs]
    if args.datasetList:
        import yaml

        configPath = Path(args.datasetList).expanduser().resolve()
        with configPath.open() as stream:
            datasets = yaml.safe_load(stream)
        if not isinstance(datasets, list):
            raise ValueError("--dataset-list expects datasetList.yaml (a list), not the top-level fit config")
        found = False
        for dataset in datasets:
            if args.dataset and dataset["name"] != args.dataset:
                continue
            found = True
            model = dataset["model"]
            for entry in model["filePathList"]:
                sources.append((entry["path"] if isinstance(entry, dict) else entry,
                                model.get("tree", args.tree)))
        if not found:
            raise ValueError(f"Dataset not found: {args.dataset}")
    resolved = []
    for rawPath, defaultTree in sources:
        rawPath = os.path.expandvars(os.path.expanduser(rawPath))
        if "$" in rawPath:
            raise ValueError(f"Unresolved environment variable in {rawPath}")
        filePath, separator, tree = rawPath.partition(".root:")
        if separator:
            filePath += ".root"
        else:
            tree = defaultTree
        if not os.path.isabs(filePath) and "://" not in filePath:
            filePath = str(Path(args.inputBase).expanduser() / filePath)
        matches = [filePath] if "://" in filePath else sorted(glob.glob(filePath))
        if not matches:
            raise FileNotFoundError(f"No input matches {filePath}")
        for match in matches:
            source = (match, tree)
            if source in resolved:
                raise ValueError(f"Duplicate input: {source}")
            resolved.append(source)
    return resolved


def loadEvents(sources, *, maxEvents=None, stepSize="100 MB", dtype="float64"):
    """Return validated host arrays and counts, retaining all non-NC events.

    The NC definition and flavor mapping follow OA2026 datasetList.yaml.
    maxEvents limits retained rows across files, in input order.
    """
    import awkward as ak
    import uproot

    chunks = {name: [] for name in ("E_GeV", "flavor_emit", "flavor_det", "isAntiNu")}
    stats = {"read": 0, "neutralCurrent": 0, "invalid": 0, "loaded": 0}
    for filePath, treeName in sources:
        if maxEvents is not None and stats["loaded"] >= maxEvents:
            break
        with uproot.open(filePath) as rootFile:
            tree = rootFile[treeName]
            for arrays in tree.iterate(["pnu", "ipnu", "OscChannel", "mode"],
                                       step_size=stepSize, library="ak"):
                energy = ak.to_numpy(ak.fill_none(ak.firsts(arrays["pnu"]), np.nan))
                pdg = ak.to_numpy(ak.fill_none(ak.firsts(arrays["ipnu"]), 0))
                channel = ak.to_numpy(arrays["OscChannel"])
                mode = np.abs(ak.to_numpy(arrays["mode"]))
                nc = (mode >= 31) & (mode <= 52)
                valid = (np.isfinite(energy) & (energy > 0) &
                         np.isin(channel, [1, 2, 3, 4, 5, 6]) &
                         np.isin(np.abs(pdg), [12, 14, 16]))
                keep = valid & ~nc
                stats["read"] += len(energy)
                stats["neutralCurrent"] += int(nc.sum())
                stats["invalid"] += int((~valid & ~nc).sum())
                indices = np.flatnonzero(keep)
                if maxEvents is not None:
                    indices = indices[:maxEvents - stats["loaded"]]
                channel = channel[indices]
                values = (energy[indices].astype(dtype),
                          np.where(np.isin(channel, [3, 4]), 0, 1).astype(np.int64),
                          np.where(np.isin(channel, [1, 2]), 1, 0).astype(np.int64),
                          pdg[indices] < 0)
                for name, value in zip(chunks, values):
                    chunks[name].append(value)
                stats["loaded"] += len(indices)
                if maxEvents is not None and stats["loaded"] >= maxEvents:
                    break
    if not stats["loaded"]:
        raise ValueError("No valid non-NC neutrino events found")
    return {name: np.concatenate(parts) for name, parts in chunks.items()}, stats


def makeEventBatch(arrays, baselineKm, device="numpy", dtype="float64"):
    """Transfer inputs once; all event arrays share the requested device."""
    from nu_waves.propagation.oscillator import NeutrinoEventBatch

    values = {"L_km": np.full(len(arrays["E_GeV"]), baselineKm, dtype=dtype), **arrays}
    if device != "numpy":
        import torch

        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in this Torch environment")
        values = {name: torch.as_tensor(value, device=target) for name, value in values.items()}
    return NeutrinoEventBatch(**values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="ROOT files/globs, optionally ending in .root:tree")
    parser.add_argument("--dataset-list", dest="datasetList", help="OA2026 datasetList.yaml")
    parser.add_argument("--dataset", help="Load only this named dataset")
    parser.add_argument("--input-base", dest="inputBase", default=".", help="Base for relative ROOT paths (default: cwd)")
    parser.add_argument("--tree", default="mtuple")
    parser.add_argument("--nu-waves-path", dest="nuWavesPath", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", default="numpy", help="numpy, cpu, cuda, cuda:0, ... (cpu uses Torch)")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--baseline-km", dest="baselineKm", type=float, default=295.)
    parser.add_argument("--max-events", dest="maxEvents", type=int)
    parser.add_argument("--step-size", dest="stepSize", default="100 MB", help="Uproot read chunk size")
    parser.add_argument("--output", help="Optional portable .npz event archive (host arrays)")
    args = parser.parse_args()
    if not args.inputs and not args.datasetList:
        parser.error("provide ROOT inputs or --dataset-list")
    if args.dataset and not args.datasetList:
        parser.error("--dataset requires --dataset-list")
    if args.maxEvents is not None and args.maxEvents <= 0:
        parser.error("--max-events must be positive")
    if not np.isfinite(args.baselineKm) or args.baselineKm <= 0:
        parser.error("--baseline-km must be finite and positive")
    configureNuWaves(args.nuWavesPath)
    import nu_waves

    sources = resolveInputs(args)
    arrays, stats = loadEvents(sources, maxEvents=args.maxEvents, stepSize=args.stepSize, dtype=args.dtype)
    events = makeEventBatch(arrays, args.baselineKm, args.device, args.dtype)
    if args.output:
        np.savez(args.output, L_km=np.full(stats["loaded"], args.baselineKm, dtype=args.dtype), **arrays)
    print(json.dumps({"nuWavesPath": nu_waves.__file__, "device": args.device,
                      "dtype": args.dtype, "inputFiles": len(sources), **stats,
                      "energyRangeGeV": [float(arrays["E_GeV"].min()), float(arrays["E_GeV"].max())],
                      "antineutrinos": int(arrays["isAntiNu"].sum())}, indent=2))
    # Return the resident batch for interactive use and future benchmark drivers.
    return events


if __name__ == "__main__":
    main()
