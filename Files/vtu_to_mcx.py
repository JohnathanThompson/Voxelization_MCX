"""
vtu_to_mcx.py
=============
Reads a SimVascular CFD result (.vtu), voxelizes the vessel geometry,
and writes an MCX-compatible JSON input file.

Tested against SimVascular output format:
  - 47 358 points / 256 452 tetrahedral cells
  - Appended, zlib-compressed, base64-encoded data arrays
  - Fields: Velocity (3-comp), Pressure, WSS (3-comp), Proc_ID

Dependencies (all standard or pip-installable):
    pip install numpy scipy tqdm

  VTK is NOT required — the reader is pure Python / NumPy.

Usage:
    python vtu_to_mcx.py \
        --input   result_050.vtu \
        --output  mcx_input.json \
        --res     0.2            \
        --pad     5              \
        --photons 1000000        \
        --scalar  Pressure       \
        --wavelength 750
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:                          # minimal no-op shim
        def __init__(self, iterable=None, **kw):
            self._it = iterable or []
        def __iter__(self):
            return iter(self._it)
        def update(self, n=1): pass
        def close(self): pass


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Pure-Python / NumPy VTU reader
#     Handles: appended encoding, base64, zlib compression
#     (exactly the format produced by SimVascular / ParaView)
# ══════════════════════════════════════════════════════════════════════════════

_VTK_DTYPE = {
    "Float32": np.float32,  "Float64": np.float64,
    "Int8":    np.int8,     "UInt8":   np.uint8,
    "Int16":   np.int16,    "UInt16":  np.uint16,
    "Int32":   np.int32,    "UInt32":  np.uint32,
    "Int64":   np.int64,    "UInt64":  np.uint64,
}


def _decode_appended_block(blob: bytes, offset: int,
                            dtype: np.dtype, n_components: int,
                            header_type: str = "UInt32",
                            compressed: bool = True) -> np.ndarray:
    """
    Decode one DataArray block from the raw (decoded) AppendedData blob.

    VTK appended + zlib layout per block
    -------------------------------------
    [hdr: num_compressed_blocks      ]  (header_type wide)
    [hdr: uncompressed_block_size    ]
    [hdr: last_partial_block_size    ]
    [hdr × num_blocks: compressed_sizes]
    [zlib-compressed data …          ]
    """
    hdr_dt   = np.dtype(_VTK_DTYPE[header_type])
    hdr_sz   = hdr_dt.itemsize
    pos      = offset

    if compressed:
        n_blocks  = int(np.frombuffer(blob[pos:pos+hdr_sz], hdr_dt)[0]); pos += hdr_sz
        _unc_sz   = int(np.frombuffer(blob[pos:pos+hdr_sz], hdr_dt)[0]); pos += hdr_sz
        _last_sz  = int(np.frombuffer(blob[pos:pos+hdr_sz], hdr_dt)[0]); pos += hdr_sz
        comp_lens = np.frombuffer(blob[pos:pos+hdr_sz*n_blocks], hdr_dt).tolist()
        pos += hdr_sz * n_blocks

        parts = []
        for cl in comp_lens:
            parts.append(zlib.decompress(blob[pos:pos+int(cl)]))
            pos += int(cl)
        raw = b"".join(parts)
    else:
        byte_len = int(np.frombuffer(blob[pos:pos+hdr_sz], hdr_dt)[0]); pos += hdr_sz
        raw = blob[pos:pos+byte_len]

    arr = np.frombuffer(raw, dtype=dtype)
    if n_components > 1:
        arr = arr.reshape(-1, n_components)
    return arr.copy()   # writable copy


def read_vtu(filepath: str) -> dict:
    """
    Parse a SimVascular .vtu file.

    Returns
    -------
    dict with keys:
        points  – (N, 3) float32   world coordinates in mm
        cells   – (M, 4) int64     tetrahedral node indices
        fields  – dict[str -> ndarray]  point-data arrays
    """
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    print(f"[INFO] Parsing {path.name} …")
    tree = ET.parse(path)
    root = tree.getroot()

    header_type = root.get("header_type", "UInt32")
    compressed  = root.get("compressor", "") != ""

    # ── AppendedData blob ────────────────────────────────────────────────────
    app_el = root.find(".//AppendedData")
    if app_el is None:
        sys.exit("[ERROR] <AppendedData> not found.")
    if app_el.get("encoding") != "base64":
        sys.exit("[ERROR] Only base64 encoding is supported.")

    raw_b64 = (app_el.text or "").strip().lstrip("_")
    blob    = base64.b64decode(raw_b64)
    print(f"[INFO] AppendedData decoded: {len(blob)/1024:.1f} KB")

    # ── DataArray metadata ───────────────────────────────────────────────────
    da_meta = {}
    for da in root.iter("DataArray"):
        da_meta[da.get("Name", "")] = {
            "dtype": np.dtype(_VTK_DTYPE.get(da.get("type","Float32"), np.float32)),
            "ncomp": int(da.get("NumberOfComponents", 1)),
            "offset": int(da.get("offset", 0)),
        }

    def load(name):
        m = da_meta[name]
        return _decode_appended_block(blob, m["offset"], m["dtype"],
                                      m["ncomp"], header_type, compressed)

    # ── Geometry ─────────────────────────────────────────────────────────────
    points       = load("Points").astype(np.float32)       # (N, 3)
    connectivity = load("connectivity").astype(np.int64)   # flat
    offsets_arr  = load("offsets").astype(np.int64)        # (M,)
    cell_types   = load("types").astype(np.uint8)          # (M,)

    # Reconstruct tetrahedral cells (VTK type 10)
    starts   = np.concatenate([[0], offsets_arr[:-1]])
    tet_idx  = np.where(cell_types == 10)[0]
    cells    = np.array([connectivity[starts[i]:offsets_arr[i]]
                         for i in tet_idx], dtype=np.int64)

    print(f"[INFO] Points: {len(points):,}  |  Tets: {len(cells):,}")

    # ── Point-data fields ────────────────────────────────────────────────────
    fields = {}
    for name in ("Velocity", "Pressure", "WSS"):
        if name in da_meta:
            fields[name] = load(name)
            print(f"[INFO] Field '{name}': shape {fields[name].shape}, "
                  f"range [{fields[name].min():.3g}, {fields[name].max():.3g}]")

    return dict(points=points, cells=cells, fields=fields)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Voxelization
# ══════════════════════════════════════════════════════════════════════════════

def voxelize(points: np.ndarray, resolution: float,
             padding: int) -> tuple:
    """
    Convert the mesh point cloud to a binary voxel volume.

    Algorithm
    ---------
    1. Build a KD-tree of all mesh nodes.
    2. For every voxel centre query the nearest node distance.
       A voxel is *inside* the vessel if that distance ≤ adaptive radius.
    3. Apply binary_fill_holes() to fill the interior (mesh nodes are
       on or near surfaces, so the lumen would otherwise be hollow).

    Parameters
    ----------
    points     : (N, 3) mesh node coordinates
    resolution : voxel edge length in the same units as points (mm)
    padding    : extra voxels to add on each face of the bounding box

    Returns
    -------
    volume : (nx, ny, nz) uint8   1 = vessel, 0 = background
    origin : (3,) float32         world coord of voxel centre [0,0,0]
    """
    pad = padding * resolution
    lo  = points.min(axis=0) - pad
    hi  = points.max(axis=0) + pad

    dims = np.ceil((hi - lo) / resolution).astype(int)
    nx, ny, nz = dims
    print(f"[INFO] Voxel grid: {nx} × {ny} × {nz}  "
          f"(res={resolution} mm, padding={padding} voxels)")

    xs = lo[0] + (np.arange(nx) + 0.5) * resolution
    ys = lo[1] + (np.arange(ny) + 0.5) * resolution
    zs = lo[2] + (np.arange(nz) + 0.5) * resolution

    # All voxel centres — shape (nx*ny*nz, 3)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centres = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    print("[INFO] Building KD-tree …")
    tree = cKDTree(points)

    # Threshold: a voxel is counted as "in tissue" if its nearest mesh
    # node is within this distance.  Tuned for typical SimVascular mesh
    # density; increase slightly (×1.2) if the surface looks porous.
    radius = resolution * np.sqrt(3) * 0.95

    print("[INFO] Running KD-tree query (parallel) …")
    dists, _ = tree.query(centres, workers=-1)

    volume = (dists <= radius).reshape(nx, ny, nz).astype(np.uint8)

    # Fill interior (lumen) voxels that the surface-node heuristic missed
    print("[INFO] Filling interior …")
    volume = binary_fill_holes(volume).astype(np.uint8)

    n_in = int(volume.sum())
    print(f"[INFO] Interior voxels: {n_in:,} / {volume.size:,} "
          f"({100*n_in/volume.size:.2f} %)")

    return volume, lo.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Map a CFD scalar field onto the voxel grid
# ══════════════════════════════════════════════════════════════════════════════

def map_scalar(points: np.ndarray, scalar: np.ndarray,
               volume: np.ndarray, origin: np.ndarray,
               resolution: float) -> np.ndarray:
    """
    Nearest-neighbour interpolation of a VTU point scalar onto voxel centres.

    Multi-component fields (Velocity, WSS) are reduced to their magnitude.

    Returns float32 array, shape == volume.shape (zeros outside vessel).
    """
    nx, ny, nz = volume.shape
    xs = origin[0] + (np.arange(nx) + 0.5) * resolution
    ys = origin[1] + (np.arange(ny) + 0.5) * resolution
    zs = origin[2] + (np.arange(nz) + 0.5) * resolution

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centres     = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    mask = volume.ravel().astype(bool)

    tree     = cKDTree(points)
    _, idx   = tree.query(centres[mask], workers=-1)

    # Reduce vector field to magnitude
    vals = scalar if scalar.ndim == 1 else np.linalg.norm(scalar, axis=1)

    out       = np.zeros(nx * ny * nz, dtype=np.float32)
    out[mask] = vals[idx].astype(np.float32)
    return out.reshape(nx, ny, nz)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Build MCX-compatible JSON
# ══════════════════════════════════════════════════════════════════════════════

# Optical properties [mua (1/mm), mus (1/mm), g, n] at common NIR wavelengths.
# Sources: Prahl's tabulated data, Jacques 2013 (Phys Med Biol).
# Label 0 = background/air, Label 1 = blood (vessel lumen + wall combined).
# Split into two labels if you want to distinguish lumen vs wall.
OPTICAL_PROPS = {
    700: {"background": [0.000, 0.000, 1.0, 1.000],
          "tissue":     [0.260, 9.800, 0.99, 1.370]},
    750: {"background": [0.000, 0.000, 1.0, 1.000],
          "tissue":     [0.230, 9.350, 0.99, 1.370]},
    800: {"background": [0.000, 0.000, 1.0, 1.000],
          "tissue":     [0.200, 8.900, 0.99, 1.370]},
    850: {"background": [0.000, 0.000, 1.0, 1.000],
          "tissue":     [0.178, 8.400, 0.99, 1.370]},
}


def build_mcx_json(volume: np.ndarray,
                   origin: np.ndarray,
                   resolution: float,
                   wavelength: int = 750,
                   nphoton: int = 1_000_000,
                   tstart: float = 0.0,
                   tend:   float = 5e-9,
                   tstep:  float = 5e-9,
                   srcpos: list | None = None,
                   srcdir: list | None = None,
                   session_id: str = "simvascular_cfd") -> tuple:
    """
    Assemble the MCX JSON dictionary and the flat binary volume.

    MCX volume convention
    ----------------------
    The binary .bin file must be uint8 values in **column-major (Fortran)
    order**: index = ix + nx*(iy + ny*iz).
    Label 0 → background, Label 1 → vessel tissue.

    Returns
    -------
    mcx_dict : dict   — ready to serialize with json.dump()
    vol_flat : ndarray uint8 (Fortran order) — write directly to .bin
    """
    nx, ny, nz = volume.shape

    # MCX uses 1-based voxel indices for source position
    if srcpos is None:
        # Default: centre of the volume, near the top face (+z)
        srcpos = [round(nx / 2) + 1,
                  round(ny / 2) + 1,
                  2]
    if srcdir is None:
        srcdir = [0.0, 0.0, 1.0]   # pencil beam along +z

    opt = OPTICAL_PROPS.get(wavelength, OPTICAL_PROPS[750])
    media = [
        # [mua, mus, g, n]
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
            "DoMismatch":   1,          # apply refractive-index mismatch BC
            "DoAutoThread": 1,          # auto-select GPU threads
            "SaveDataMask": "dpsp",     # fluence | detected photons | scatter | partial path
            "OutputFormat": "jnii",     # JNIfTI output (or use "hdr" for Analyze)
            "OutputType":   "X",        # output fluence (X = normalized fluence rate)
        },
        "Forward": {
            "T0": tstart,
            "T1": tend,
            "Dt": tstep,
        },
        "Optode": {
            "Source": {
                "Type":    "pencil",    # change to "disk", "gaussian", etc. as needed
                "Pos":     srcpos,      # 1-indexed voxel [x, y, z]
                "Dir":     srcdir,      # unit direction vector
                "Param1":  [0.0, 0.0, 0.0, 0.0],
                "Param2":  [0.0, 0.0, 0.0, 0.0],
            },
            # Add detector entries here, e.g.:
            # {"Pos": [x, y, z], "R": radius_mm}
            "Detector": []
        },
        "Domain": {
            "OriginType": 1,             # 1 = origin at centre of voxel (0,0,0)
            "LengthUnit": resolution,    # voxel edge length in mm
            "Media":      media,
            "Dim":        [nx, ny, nz],
            "VolumeFile": f"{session_id}.bin",  # path to binary volume file
            # If you want inline base64 volume instead of a .bin file,
            # uncomment the next line and comment out "VolumeFile":
            # "Vol": base64.b64encode(volume.flatten(order='F').tobytes()).decode()
        },
        # Non-MCX metadata (ignored by the solver, useful for reproducibility)
        "_metadata": {
            "origin_world_mm":  origin.tolist(),
            "resolution_mm":    resolution,
            "grid_dims":        [nx, ny, nz],
            "wavelength_nm":    wavelength,
            "n_inside_voxels":  int(volume.sum()),
            "source":           "SimVascular CFD voxelization",
        }
    }

    vol_flat = volume.flatten(order='F').astype(np.uint8)
    return mcx, vol_flat


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Voxelize a SimVascular CFD .vtu file and export to MCX JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--input",  "-i", required=True,
                   help="SimVascular result .vtu file")
    p.add_argument("--output", "-o", default="mcx_input.json",
                   help="Output JSON filename")
    p.add_argument("--res",    "-r", type=float, default=0.2,
                   help="Voxel resolution in mm")
    p.add_argument("--pad",    "-p", type=int,   default=5,
                   help="Padding voxels around bounding box")
    p.add_argument("--photons",      type=int,   default=1_000_000,
                   help="Number of photon packets for MCX")
    p.add_argument("--wavelength",   type=int,   default=750,
                   choices=[700, 750, 800, 850],
                   help="Wavelength (nm) for optical property lookup")
    p.add_argument("--scalar",       default="Pressure",
                   choices=["Pressure", "Velocity", "WSS"],
                   help="CFD scalar to export as a companion .npy file")
    p.add_argument("--srcpos",       nargs=3, type=float, default=None,
                   metavar=("X","Y","Z"),
                   help="MCX source position in 1-indexed voxel coords")
    p.add_argument("--srcdir",       nargs=3, type=float, default=None,
                   metavar=("DX","DY","DZ"),
                   help="MCX source direction unit vector")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load VTU ───────────────────────────────────────────────────────────
    data   = read_vtu(args.input)
    points = data["points"]
    fields = data["fields"]

    # ── 2. Voxelize ───────────────────────────────────────────────────────────
    volume, origin = voxelize(points, args.res, args.pad)

    # ── 3. Export companion CFD scalar (optional) ─────────────────────────────
    if args.scalar in fields:
        print(f"[INFO] Mapping scalar '{args.scalar}' onto voxel grid …")
        sv = map_scalar(points, fields[args.scalar], volume, origin, args.res)
        npy_path = Path(args.output).with_suffix("") \
                   .with_name(Path(args.output).stem + f"_{args.scalar.lower()}.npy")
        np.save(str(npy_path), sv)
        print(f"[INFO] Scalar map saved → {npy_path}")
    else:
        print(f"[WARN] Scalar '{args.scalar}' not in VTU; skipping.")

    # ── 4. Assemble MCX JSON ──────────────────────────────────────────────────
    session_id = Path(args.input).stem
    srcpos = [float(v) for v in args.srcpos] if args.srcpos else None
    srcdir = [float(v) for v in args.srcdir] if args.srcdir else None

    mcx_dict, vol_flat = build_mcx_json(
        volume, origin, args.res,
        wavelength = args.wavelength,
        nphoton    = args.photons,
        srcpos     = srcpos,
        srcdir     = srcdir,
        session_id = session_id,
    )

    # ── 5. Write JSON + binary volume ─────────────────────────────────────────
    out_json = Path(args.output)
    out_bin  = out_json.with_suffix(".bin")

    with open(out_json, "w") as f:
        json.dump(mcx_dict, f, indent=2)
    print(f"[INFO] MCX JSON    → {out_json}  ({out_json.stat().st_size/1024:.1f} KB)")

    vol_flat.tofile(str(out_bin))
    print(f"[INFO] Volume .bin → {out_bin}  ({out_bin.stat().st_size/1024:.1f} KB)")

    # ── 6. Print the MCX run command ──────────────────────────────────────────
    nx, ny, nz = volume.shape
    print()
    print("═" * 65)
    print("  Run MCX with the command below (adjust --gpu as needed):")
    print()
    print(f"    mcx -f {out_json} \\")
    print(f"        --vol {out_bin} \\")
    print(f"        --dim {nx},{ny},{nz} \\")
    print(f"        --gpu 1")
    print("═" * 65)
    print()
    print("  Tip: if your MCX version supports JSON-embedded volumes,")
    print("  uncomment the '\"Vol\"' line in build_mcx_json() and remove")
    print("  the --vol / --dim flags from the command above.")


if __name__ == "__main__":
    main()
