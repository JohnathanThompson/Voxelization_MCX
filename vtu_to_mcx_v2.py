"""
vtu_to_mcx.py
=============
Reads a SimVascular CFD result (.vtu), voxelizes the vessel geometry,
and writes an MCX-compatible JSON + binary volume file.

Validated against SimVascular output (result_050.vtu):
  47 358 points | 256 452 tetrahedra
  Fields: Velocity (3-comp), Pressure, WSS (3-comp)
  Encoding: appended base64 + zlib, LittleEndian, UInt32 headers

Install dependencies:
    pip install numpy scipy meshio

meshio is used when available (simpler). If not installed, a built-in
pure-Python reader handles the exact SimVascular base64/zlib format.

Usage:
    python vtu_to_mcx.py --input result_050.vtu --output mcx_input.json
    python vtu_to_mcx.py --input result_050.vtu --output mcx_input.json --res 0.25 --pad 4
    python vtu_to_mcx.py --input result_050.vtu --output mcx_input.json --res 0.2 --wavelength 850
"""

import argparse
import base64
import json
import sys
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree


# ══════════════════════════════════════════════════════════════════════════════
# 1.  VTU READER
#     Tries meshio first (pip install meshio), falls back to built-in decoder.
# ══════════════════════════════════════════════════════════════════════════════

def read_vtu(filepath: str) -> dict:
    """
    Returns dict with keys:
        points  – (N, 3) float32  node coordinates in mm
        fields  – dict[str -> ndarray]  point-data arrays
                  Velocity: (N,3), Pressure: (N,), WSS: (N,3)
    """
    try:
        import meshio
        return _read_meshio(filepath)
    except ImportError:
        print("[INFO] meshio not found — using built-in VTU reader.")
        print("       Install with:  pip install meshio")
        return _read_builtin(filepath)


def _read_meshio(filepath: str) -> dict:
    import meshio
    print(f"[INFO] Reading {Path(filepath).name} with meshio …")
    mesh = meshio.read(filepath)
    points = mesh.points.astype(np.float32)
    fields = {}
    for name, arr in mesh.point_data.items():
        fields[name] = arr
        print(f"[INFO] Field '{name}': shape {arr.shape}, "
              f"range [{arr.min():.4g}, {arr.max():.4g}]")
    print(f"[INFO] Points: {len(points):,}")
    return dict(points=points, fields=fields)


# ─── Built-in reader ─────────────────────────────────────────────────────────
# Handles SimVascular's specific encoding:
#   AppendedData base64, vtkZLibDataCompressor, UInt32 headers, LittleEndian.
#
# Each DataArray block in the file uses one of two layouts:
#   (A) Two-part: base64(header) + base64(all_compressed_chunks_concatenated)
#       Identified by a '=' padding char appearing before the last 4 chars.
#   (B) One-part: base64(header + all_compressed_chunks_concatenated)
#       No mid-stream '='.
# The byte offsets in the XML are char offsets from the '_' marker.
# Block boundaries = consecutive offset values; last block ends at </AppendedData>.

_VTK_DTYPE = {
    "Float32": np.float32, "Float64": np.float64,
    "Int8":    np.int8,    "UInt8":   np.uint8,
    "Int16":   np.int16,   "UInt16":  np.uint16,
    "Int32":   np.int32,   "UInt32":  np.uint32,
    "Int64":   np.int64,   "UInt64":  np.uint64,
}


def _decode_block(raw_file: bytes, data_start: int, c0: int, c1: int) -> bytes:
    """
    Decode and decompress one DataArray block from the raw file bytes.
    c0, c1: char offsets from data_start that delimit this block.
    """
    seg = raw_file[data_start + c0 : data_start + c1].rstrip(b" \t\n\r")

    # Detect two-part layout: a '=' before the last 4 chars
    mid_eq = next((i for i in range(len(seg) - 4) if seg[i] == ord("=")), None)

    if mid_eq is not None:
        # Part 1: header encoded independently
        hdr_end = ((mid_eq + 4) // 4) * 4
        hdr = base64.b64decode(seg[:hdr_end])
        n_b    = int.from_bytes(hdr[0:4], "little")
        csizes = [int.from_bytes(hdr[12 + i*4 : 16 + i*4], "little") for i in range(n_b)]
        # Part 2: all compressed chunks in one b64 blob
        tail = seg[hdr_end:]
        pad  = (4 - len(tail) % 4) % 4
        data_raw = base64.b64decode(tail + b"=" * pad)
    else:
        # Single b64 blob: header bytes followed by compressed chunks
        pad     = (4 - len(seg) % 4) % 4
        all_dec = base64.b64decode(seg + b"=" * pad)
        n_b     = int.from_bytes(all_dec[0:4], "little")
        n_hdr   = (3 + n_b) * 4
        csizes  = [int.from_bytes(all_dec[12 + i*4 : 16 + i*4], "little") for i in range(n_b)]
        data_raw = all_dec[n_hdr:]

    pos, parts = 0, []
    for cs in csizes:
        parts.append(zlib.decompress(data_raw[pos : pos + cs]))
        pos += cs
    return b"".join(parts)


def _read_builtin(filepath: str) -> dict:
    print(f"[INFO] Reading {Path(filepath).name} with built-in reader …")
    with open(filepath, "rb") as f:
        raw = f.read()

    # Locate the data blob (starts one byte after the '_' marker)
    app_start = raw.find(b"<AppendedData")
    us        = raw.find(b"_", app_start)
    app_end   = raw.find(b"</AppendedData>", us)
    data_start = us + 1

    # Parse XML metadata for each DataArray
    tree = ET.parse(filepath)
    root = tree.getroot()
    das  = []
    for da in root.iter("DataArray"):
        das.append({
            "name":   da.get("Name", ""),
            "dtype":  _VTK_DTYPE.get(da.get("type", "Float32"), np.float32),
            "ncomp":  int(da.get("NumberOfComponents", 1)),
            "offset": int(da.get("offset", 0)),
        })
    # Set char-end for each block
    for i in range(len(das) - 1):
        das[i]["end"] = das[i + 1]["offset"]
    das[-1]["end"] = app_end - data_start

    arrays = {}
    for da in das:
        raw_bytes = _decode_block(raw, data_start, da["offset"], da["end"])
        arr = np.frombuffer(raw_bytes, dtype=da["dtype"])
        if da["ncomp"] > 1:
            arr = arr.reshape(-1, da["ncomp"])
        arrays[da["name"]] = arr
        print(f"[INFO] Field '{da['name']}': shape {arr.shape}, "
              f"range [{arr.min():.4g}, {arr.max():.4g}]")

    points = arrays.pop("Points").astype(np.float32)
    # Remove cell/topology arrays — keep only point-data fields
    fields = {k: v for k, v in arrays.items()
              if k in ("Velocity", "Pressure", "WSS")}
    print(f"[INFO] Points: {len(points):,}")
    return dict(points=points, fields=fields)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  VOXELIZATION
# ══════════════════════════════════════════════════════════════════════════════

def voxelize(points: np.ndarray, resolution: float, padding: int) -> tuple:
    """
    Convert mesh node cloud → binary voxel grid.

    Algorithm
    ---------
    1. Build KD-tree on all mesh nodes.
    2. Mark voxels whose centre lies within `radius` of any mesh node.
       (Nodes sit on the vessel surface, so this captures the shell.)
    3. binary_fill_holes() fills the interior lumen.

    Returns
    -------
    volume : (nx, ny, nz) uint8   1 = vessel, 0 = background
    origin : (3,) float32         world coordinate of voxel centre [0,0,0]
    """
    pad = padding * resolution
    lo  = points.min(axis=0) - pad
    hi  = points.max(axis=0) + pad
    nx, ny, nz = np.ceil((hi - lo) / resolution).astype(int)

    print(f"[INFO] Voxel grid: {nx} × {ny} × {nz}  "
          f"(res={resolution} mm, pad={padding} voxels)")

    xs = lo[0] + (np.arange(nx) + 0.5) * resolution
    ys = lo[1] + (np.arange(ny) + 0.5) * resolution
    zs = lo[2] + (np.arange(nz) + 0.5) * resolution
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centres = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    print("[INFO] Querying KD-tree …")
    kd    = cKDTree(points)
    # Threshold: sqrt(3)*res covers the full voxel diagonal
    # ×0.9 avoids over-expansion at thin walls
    radius = resolution * np.sqrt(3) * 0.9
    dists, _ = kd.query(centres, workers=-1)

    volume = (dists <= radius).reshape(nx, ny, nz).astype(np.uint8)

    print("[INFO] Filling interior …")
    volume = binary_fill_holes(volume).astype(np.uint8)

    n_in = int(volume.sum())
    print(f"[INFO] Inside voxels: {n_in:,} / {volume.size:,} "
          f"({100 * n_in / volume.size:.2f} %)")
    return volume, lo.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CFD SCALAR → VOXEL GRID  (optional companion export)
# ══════════════════════════════════════════════════════════════════════════════

def map_scalar(points: np.ndarray, scalar: np.ndarray,
               volume: np.ndarray, origin: np.ndarray,
               resolution: float) -> np.ndarray:
    """
    Nearest-neighbour interpolation of a VTU point field onto voxel centres.
    Multi-component fields (Velocity, WSS) are reduced to their magnitude.
    Returns float32 array, same shape as volume (zeros outside vessel).
    """
    nx, ny, nz = volume.shape
    xs = origin[0] + (np.arange(nx) + 0.5) * resolution
    ys = origin[1] + (np.arange(ny) + 0.5) * resolution
    zs = origin[2] + (np.arange(nz) + 0.5) * resolution
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centres = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    mask = volume.ravel().astype(bool)
    kd   = cKDTree(points)
    _, idx = kd.query(centres[mask], workers=-1)

    vals = scalar if scalar.ndim == 1 else np.linalg.norm(scalar, axis=1)
    out  = np.zeros(nx * ny * nz, dtype=np.float32)
    out[mask] = vals[idx].astype(np.float32)
    return out.reshape(nx, ny, nz)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MCX JSON ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

# Optical properties [mua (1/mm), mus (1/mm), g, n] for blood at NIR wavelengths.
# Label 0 = air/background, Label 1 = blood vessel (lumen + wall combined).
# Source: Prahl tabulated data; Jacques 2013 Phys Med Biol.
# Adjust these for your specific tissue composition and wavelength.
OPTICAL_PROPS = {
    700: {"background": [0.000, 0.000, 1.00, 1.000],
          "tissue":     [0.260, 9.800, 0.99, 1.370]},
    750: {"background": [0.000, 0.000, 1.00, 1.000],
          "tissue":     [0.230, 9.350, 0.99, 1.370]},
    800: {"background": [0.000, 0.000, 1.00, 1.000],
          "tissue":     [0.200, 8.900, 0.99, 1.370]},
    850: {"background": [0.000, 0.000, 1.00, 1.000],
          "tissue":     [0.178, 8.400, 0.99, 1.370]},
}


def build_mcx_json(volume: np.ndarray,
                   origin: np.ndarray,
                   resolution: float,
                   session_id: str,
                   bin_filename: str,
                   wavelength: int = 750,
                   nphoton: int = 1_000_000,
                   tstart: float = 0.0,
                   tend:   float = 5e-9,
                   tstep:  float = 5e-9,
                   srcpos: list | None = None,
                   srcdir: list | None = None) -> tuple:
    """
    Build the MCX JSON dict and the flat uint8 volume (Fortran order).

    MCX volume convention
    ----------------------
    The .bin file must be uint8, flattened in column-major (Fortran) order:
      index = ix + nx*(iy + ny*iz)
    Label 0 → background/air, Label 1 → vessel tissue.

    Returns (mcx_dict, vol_flat_uint8)
    """
    nx, ny, nz = volume.shape

    # Default source: centre of XY, near the -Z face, pointing +Z
    if srcpos is None:
        srcpos = [int(nx // 2) + 1, int(ny // 2) + 1, 2]   # 1-indexed for MCX
    if srcdir is None:
        srcdir = [0.0, 0.0, 1.0]

    opt = OPTICAL_PROPS.get(wavelength, OPTICAL_PROPS[750])
    media = [
        {"mua": opt["background"][0], "mus": opt["background"][1],
         "g":   opt["background"][2], "n":   opt["background"][3]},
        {"mua": opt["tissue"][0],     "mus": opt["tissue"][1],
         "g":   opt["tissue"][2],     "n":   opt["tissue"][3]},
    ]

    mcx = {
        "Session": {
            "ID":           session_id,
            "Photons":      nphoton,
            "Seed":         29012392,
            "DoMismatch":   1,        # apply refractive-index mismatch BC
            "DoAutoThread": 1,        # auto-select GPU threads
            "SaveDataMask": "dpsp",   # fluence | detected photons | scatter | partial path
            "OutputFormat": "jnii",   # JNIfTI (or use "hdr" for Analyze/NIfTI)
            "OutputType":   "X",      # normalized fluence rate
        },
        "Forward": {
            "T0": tstart,
            "T1": tend,
            "Dt": tstep,
        },
        "Optode": {
            "Source": {
                "Type":   "pencil",   # pencil | disk | gaussian | cone | …
                "Pos":    srcpos,     # 1-indexed voxel [x, y, z]
                "Dir":    srcdir,     # unit direction vector
                "Param1": [0.0, 0.0, 0.0, 0.0],
                "Param2": [0.0, 0.0, 0.0, 0.0],
            },
            # Add detector dicts here as needed, e.g.:
            # {"Pos": [x, y, z], "R": radius_mm}
            "Detector": [],
        },
        "Domain": {
            "OriginType": 1,            # 1 = origin at centre of voxel (0,0,0)
            "LengthUnit": resolution,   # voxel edge length in mm
            "Media":      media,
            "Dim":        [nx, ny, nz],
            "VolumeFile": bin_filename, # path to binary volume (.bin)
        },
        # Non-MCX metadata — ignored by the solver, useful for records
        "_metadata": {
            "origin_world_mm":  origin.tolist(),
            "resolution_mm":    resolution,
            "grid_dims":        [nx, ny, nz],
            "wavelength_nm":    wavelength,
            "n_vessel_voxels":  int(volume.sum()),
            "source_file":      "SimVascular CFD voxelization",
        },
    }

    vol_flat = volume.flatten(order="F").astype(np.uint8)
    return mcx, vol_flat


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Voxelize a SimVascular CFD .vtu and export to MCX JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",  "-i", required=True,
                   help="SimVascular result .vtu file")
    p.add_argument("--output", "-o", default="mcx_input.json",
                   help="Output JSON filename")
    p.add_argument("--res",    "-r", type=float, default=0.25,
                   help="Voxel resolution in mm")
    p.add_argument("--pad",    "-p", type=int,   default=4,
                   help="Padding voxels around bounding box on each face")
    p.add_argument("--photons",      type=int,   default=1_000_000,
                   help="Number of MCX photon packets")
    p.add_argument("--wavelength",   type=int,   default=750,
                   choices=[700, 750, 800, 850],
                   help="Wavelength (nm) — selects optical property preset")
    p.add_argument("--scalar",       default=None,
                   choices=["Pressure", "Velocity", "WSS"],
                   help="Export a CFD scalar as a companion .npy file")
    p.add_argument("--srcpos", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"),
                   help="MCX source position in 1-indexed voxel coords")
    p.add_argument("--srcdir", nargs=3, type=float, default=None,
                   metavar=("DX", "DY", "DZ"),
                   help="MCX source direction unit vector")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load VTU ──────────────────────────────────────────────────────────
    data   = read_vtu(args.input)
    points = data["points"]
    fields = data["fields"]

    # ── 2. Voxelize ──────────────────────────────────────────────────────────
    volume, origin = voxelize(points, args.res, args.pad)

    # ── 3. Optional: export a CFD scalar as .npy ─────────────────────────────
    if args.scalar:
        if args.scalar in fields:
            print(f"[INFO] Mapping '{args.scalar}' onto voxel grid …")
            sv = map_scalar(points, fields[args.scalar], volume, origin, args.res)
            npy_path = Path(args.output).with_suffix("") \
                           .with_name(Path(args.output).stem + f"_{args.scalar.lower()}.npy")
            np.save(str(npy_path), sv)
            print(f"[INFO] Scalar map saved → {npy_path}")
        else:
            print(f"[WARN] Field '{args.scalar}' not found in VTU; skipping.")

    # ── 4. Build MCX JSON ─────────────────────────────────────────────────────
    session_id   = Path(args.input).stem
    out_json     = Path(args.output)
    out_bin      = out_json.with_suffix(".bin")
    bin_filename = out_bin.name   # relative path for portability

    srcpos = [float(v) for v in args.srcpos] if args.srcpos else None
    srcdir = [float(v) for v in args.srcdir] if args.srcdir else None

    mcx_dict, vol_flat = build_mcx_json(
        volume, origin, args.res, session_id, bin_filename,
        wavelength=args.wavelength,
        nphoton=args.photons,
        srcpos=srcpos,
        srcdir=srcdir,
    )

    # ── 5. Write outputs ──────────────────────────────────────────────────────
    with open(out_json, "w") as f:
        json.dump(mcx_dict, f, indent=2)
    print(f"[INFO] MCX JSON    → {out_json}  ({out_json.stat().st_size / 1024:.1f} KB)")

    vol_flat.tofile(str(out_bin))
    print(f"[INFO] Volume .bin → {out_bin}  ({out_bin.stat().st_size / 1024:.1f} KB)")

    # ── 6. Print run command ──────────────────────────────────────────────────
    nx, ny, nz = volume.shape
    print()
    print("═" * 62)
    print("  Run MCX:")
    print()
    print(f"    mcx -f {out_json.name} \\")
    print(f"        --vol {out_bin.name} \\")
    print(f"        --dim {nx},{ny},{nz} \\")
    print(f"        --gpu 1")
    print()
    print("  Source is placed at the centre of the volume near the -Z face.")
    print("  Adjust --srcpos / --srcdir to match your optical setup.")
    print("═" * 62)


if __name__ == "__main__":
    main()
