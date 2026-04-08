"""
extract_vessel_absorption.py
=============================
Extract and visualize absorption values from MCX output JNIfTI.
Optionally mask to vessel region if binary volume is provided.
Creates matplotlib slices and histogram for direct visualization in Python/MATLAB.

Usage:
    # Just visualize MCX output (no vessel mask needed):
    python extract_vessel_absorption.py --jnii result_0501.jnii --plot
    
    # With optional vessel mask (must match JNII dims):
    python extract_vessel_absorption.py \\
        --jnii result_0501.jnii \\
        --volume mcx_input_3.bin \\
        --dims 60,60 \\
        --plot
"""

import argparse
import json
import numpy as np
import zlib
import base64
from pathlib import Path
import datetime


def load_jnii(jnii_path):
    """Load JNIfTI file and extract absorption data."""
    with open(jnii_path, 'r') as f:
        data = json.load(f)
    
    nifti_data = data.get('NIFTIData', {})
    
    # Extract compressed array data
    compressed = base64.b64decode(nifti_data['_ArrayZipData_'])
    uncompressed = zlib.decompress(compressed)
    
    # Get stated dimensions from header
    dims = tuple(nifti_data['_ArraySize_'][:3])  # Take only first 3 (x,y,z)
    expected_size = np.prod(dims) * 4  # float32 = 4 bytes
    actual_size = len(uncompressed)
    n_elements = actual_size // 4
    
    print(f"[INFO] JNII header dims: {dims}")
    print(f"[INFO] Expected data size: {expected_size} bytes, actual: {actual_size} bytes")
    print(f"[INFO] Total elements: {n_elements}")
    
    # If size mismatch, try to infer actual dims
    if expected_size != actual_size:
        print(f"[WARN] Size mismatch! Attempting to infer dimensions...")
        
        # Try cube first
        cube_size = round(n_elements ** (1/3))
        if (cube_size ** 3) == n_elements:
            dims = (cube_size, cube_size, cube_size)
            print(f"[INFO] Inferred cubic dims: {dims}")
        # Try keeping first two dims and extending third (common for time series)
        elif n_elements % (dims[0] * dims[1]) == 0:
            new_z = n_elements // (dims[0] * dims[1])
            dims = (dims[0], dims[1], new_z)
            print(f"[INFO] Inferred dims (keeping X,Y): {dims}")
        # Try factorization with ratio close to header
        elif n_elements % np.prod(dims) == 0:
            ratio = n_elements // np.prod(dims)
            dims = (dims[0], dims[1], dims[2] * ratio)
            print(f"[INFO] Inferred dims (time-extended): {dims}")
        else:
            print(f"[ERROR] Cannot factorize {n_elements} elements to match data structure")
            raise ValueError(f"Data size {actual_size} bytes ({n_elements} elements) "
                           f"does not match header {expected_size} bytes {np.prod(dims)} elements.")
    
    # Reshape and load
    try:
        arr = np.frombuffer(uncompressed, dtype=np.float32).reshape(dims)
        print(f"[INFO] Reshaped to {arr.shape}")
    except ValueError as e:
        print(f"[ERROR] Reshape to {dims} failed: {e}")
        raise
    
    print(f"[INFO] Loaded JNIfTI: shape {arr.shape}, "
          f"range [{arr.min():.4e}, {arr.max():.4e}]")
    
    return arr, dims


def load_volume(bin_path, dims):
    """Load binary volume file (voxel labels)."""
    vol = np.fromfile(bin_path, dtype=np.uint8).reshape(dims, order='F')
    n_vessel = int((vol == 1).sum())
    print(f"[INFO] Loaded volume: shape {vol.shape}, "
          f"vessel voxels: {n_vessel:,} / {vol.size:,}")
    return vol


def extract_vessel_absorption(absorption, vessel_mask):
    """Extract absorption values only where vessel exists (mask==1)."""
    vessel_absorption = np.zeros_like(absorption)
    vessel_absorption[vessel_mask == 1] = absorption[vessel_mask == 1]
    
    vessel_vals = absorption[vessel_mask == 1]
    print(f"[INFO] Vessel absorption: "
          f"mean={vessel_vals.mean():.4e}, "
          f"max={vessel_vals.max():.4e}, "
          f"min={vessel_vals.min():.4e}")
    
    return vessel_absorption


def main():
    p = argparse.ArgumentParser(
        description="Extract and visualize vessel absorption from MCX JNIfTI output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--jnii", required=True,
                   help="MCX output JNIfTI file (result_*.jnii)")
    p.add_argument("--volume", default=None,
                   help="Binary volume file (*.bin) — optional, must match JNII dims")
    p.add_argument("--dims", default=None,
                   help="Volume dimensions as 'nx,ny,nz' (auto-detected from JNII if omitted)")
    p.add_argument("--output", default="vessel_absorption.npy",
                   help="Output .npy file for absorption map")
    p.add_argument("--plot", action="store_true",
                   help="Generate matplotlib slices and histogram")
    p.add_argument("--plotdir", default="images",
                   help="Folder to save the plotted PNG")
    
    args = p.parse_args()
    
    # Load JNII (auto-detects dims)
    print(f"[INFO] Loading JNII from {args.jnii}")
    absorption, jnii_dims = load_jnii(args.jnii)
    
    # Load optional volume mask
    vessel_mask = None
    if args.volume:
        # Determine dims for volume: use provided arg, JNII dims, or try to load
        if args.dims:
            dims = tuple(int(d) for d in args.dims.split(","))
        else:
            dims = jnii_dims
            print(f"[INFO] Using JNII dims for volume: {dims}")
        
        try:
            vessel_mask = load_volume(args.volume, dims)
            if vessel_mask.shape != absorption.shape:
                print(f"[WARN] Shape mismatch: volume {vessel_mask.shape} vs absorption {absorption.shape}")
                print(f"[WARN] Attempting to use volume anyway (will warn on extract)...")
        except Exception as e:
            print(f"[WARN] Could not load volume: {e}")
            print(f"[WARN] Skipping vessel mask — will visualize full absorption field")
            vessel_mask = None
    
    # Extract or use full absorption
    if vessel_mask is not None:
        try:
            result = extract_vessel_absorption(absorption, vessel_mask)
        except Exception as e:
            print(f"[WARN] Could not extract vessel absorption: {e}")
            print(f"[WARN] Using full absorption field instead")
            result = absorption
    else:
        result = absorption
        print(f"[INFO] Using full absorption field (no masking)")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), result)
    print(f"[INFO] Saved -> {output_path}")
    
    # Plot if requested
    if args.plot:
        plot_absorption(result, output_path, Path(args.plotdir))
    
    print()
    print("="*60)
    print(f"  Absorption range: [{result.min():.4e}, {result.max():.4e}]")
    print(f"  Mean: {result[result > 0].mean():.4e}")
    print("="*60)


def plot_absorption(data, label_path: Path, plot_dir: Path):
    """Create matplotlib visualization of absorption."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plots")
        print("       Install with: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"MCX Absorption: {label_path.stem}", fontsize=14)
    
    # Three orthogonal slices through center
    nx, ny, nz = data.shape
    cx, cy, cz = nx//2, ny//2, nz//2
    
    # XY slice (z=center)
    ax = axes[0, 0]
    im = ax.imshow(data[:, :, cz].T, cmap='hot', origin='lower')
    ax.set_title(f'XY-plane (z={cz})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Absorption')
    
    # XZ slice (y=center)
    ax = axes[0, 1]
    im = ax.imshow(data[:, cy, :].T, cmap='hot', origin='lower')
    ax.set_title(f'XZ-plane (y={cy})')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.colorbar(im, ax=ax, label='Absorption')
    
    # YZ slice (x=center)
    ax = axes[1, 0]
    im = ax.imshow(data[cx, :, :].T, cmap='hot', origin='lower')
    ax.set_title(f'YZ-plane (x={cx})')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    plt.colorbar(im, ax=ax, label='Absorption')
    
    # Histogram
    ax = axes[1, 1]
    vals = data[data > 0].ravel()  # Exclude zero/background
    ax.hist(vals, bins=100, color='darkred', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Absorption')
    ax.set_ylabel('Voxel count')
    ax.set_title('Absorption histogram (non-zero)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = plot_dir / f"{label_path.stem}_{timestamp}.png"
    plt.savefig(str(plot_path), dpi=100, bbox_inches='tight')
    print(f"[INFO] Plot saved -> {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
