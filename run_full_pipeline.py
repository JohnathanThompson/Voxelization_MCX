#!/usr/bin/env python3
"""
run_full_pipeline.py
====================
All-in-one script to run the complete MCX pipeline:
1. Generate MCX input from VTU using vtu_to_mcx_v2.py
2. Run MCX simulation
3. Extract and visualize absorption with timestamped PNG output

Usage:
    python run_full_pipeline.py --input result_050.vtu
"""

import argparse
import subprocess
from pathlib import Path
import os


def main():
    p = argparse.ArgumentParser(
        description="Run complete MCX pipeline: VTU → MCX input → Simulation → Absorption visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", "-i", required=True,
                   help="Input SimVascular VTU file (e.g., result_050.vtu)")
    p.add_argument("--res", "-r", type=float, default=0.2,
                   help="Voxel resolution in mm")
    p.add_argument("--pad", "-p", type=int, default=4,
                   help="Padding voxels around bounding box")
    p.add_argument("--photons", type=int, default=10_000_000,
                   help="Number of MCX photons")
    p.add_argument("--wavelength", type=int, default=750,
                   choices=[700, 750, 800, 850],
                   help="Wavelength (nm) for optical properties")

    args = p.parse_args()

    vtu_path = Path(args.input)
    if not vtu_path.exists():
        raise FileNotFoundError(f"Input VTU file not found: {vtu_path}")

    session_id = vtu_path.stem

    # Create output directories
    os.makedirs("mcx_input", exist_ok=True)
    os.makedirs("mcx_output", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    json_path = Path(f"mcx_input/{session_id}.json")
    bin_path = Path(f"mcx_input/{session_id}.bin")
    jnii_path = Path(f"mcx_output/{session_id}.jnii")
    npy_path = Path(f"mcx_output/{session_id}_absorption.npy")

    print("=" * 60)
    print("MCX Pipeline: VTU → MCX input → Simulation → Absorption visualization")
    print("=" * 60)

    # Step 1: Generate MCX input
    print("\n[1/3] Generating MCX input from VTU...")
    cmd1 = [
        "python", "vtu_to_mcx_v2.py",
        "--input", str(vtu_path),
        "--output", str(json_path),
        "--res", str(args.res),
        "--pad", str(args.pad),
        "--photons", str(args.photons),
        "--wavelength", str(args.wavelength)
    ]
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    if result1.returncode != 0:
        print(f"[ERROR] Failed to generate MCX input:\n{result1.stderr}")
        return
    print(result1.stdout.strip())

    # Step 2: Run MCX simulation
    print("\n[2/3] Running MCX simulation...")
    mcx_exe = Path("mcx-win-x86_64-v2025.10/mcx/bin/mcx.exe")
    if not mcx_exe.exists():
        # Try relative path or assume in PATH
        mcx_exe = "mcx"
    cmd2 = [str(mcx_exe), "-f", str(json_path.resolve()), "-A"]
    result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=str(Path("mcx_output").resolve()))
    if result2.returncode != 0:
        print(f"[ERROR] MCX simulation failed:\n{result2.stderr}")
        return
    print("Simulation completed successfully.")
    # Extract key output
    lines = result2.stdout.split('\n')
    for line in lines:
        if 'total simulated energy:' in line or 'absorbed:' in line:
            print(line.strip())

    # Step 3: Extract and visualize absorption
    print("\n[3/3] Extracting and visualizing absorption...")
    cmd3 = [
        "python", "extract_vessel_absorption.py",
        "--jnii", str(jnii_path),
        "--output", str(npy_path),
        "--plot",
        "--plotdir", "images"
    ]
    result3 = subprocess.run(cmd3, capture_output=True, text=True)
    if result3.returncode != 0:
        print(f"[ERROR] Absorption extraction failed:\n{result3.stderr}")
        return
    print(result3.stdout.strip())

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Output files:\n  MCX input: {json_path}, {bin_path}\n  MCX output: {jnii_path}, {npy_path}\n  Plot: images/{session_id}_absorption_<timestamp>.png")
    print("=" * 60)


if __name__ == "__main__":
    main()