"""Inspect live Python objects in the current process and report estimated memory usage.

This script must be run inside the Python process you'd like to inspect (for example,
run it from a REPL in which your application is running, or insert `import tools.mem_inspect
; mem_inspect.snapshot()` in the code at the point you want to sample).

Usage (standalone):
  python tools/mem_inspect.py --top 20

Usage (from code):
  from tools.mem_inspect import snapshot
  snapshot(top=20)

The script reports:
- Top object types by estimated bytes
- Top instances for numpy arrays, PIL Images and generic objects (repr preview)

Limitations:
- `sys.getsizeof` does not report real memory used by some objects (e.g., numpy arrays' buffer),
  so we treat numpy arrays specially using `.nbytes`.
- Pygame Surfaces and some C-level objects are estimated where possible.
- To capture Python allocation traces, run the process with `tracemalloc.start()` early and
  use the `--tracemalloc` flag (if desired).
"""
from collections import defaultdict, Counter
import gc
import sys
import argparse
import json
import types

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pygame
except Exception:
    pygame = None

import inspect


def estimate_obj_size(obj):
    """Estimate memory usage (in bytes) for an object where possible."""
    # numpy arrays: use nbytes
    try:
        if np is not None and isinstance(obj, np.ndarray):
            return int(obj.nbytes)
    except Exception:
        pass
    # PIL Image
    try:
        if Image is not None and isinstance(obj, Image.Image):
            w, h = obj.size
            mode = obj.mode
            channels = 1 if mode == 'L' else 3
            return int(w * h * channels)
    except Exception:
        pass
    # pygame Surface
    try:
        if pygame is not None and isinstance(obj, pygame.Surface):
            try:
                bpp = obj.get_bytesize()
                w, h = obj.get_size()
                return int(w * h * bpp)
            except Exception:
                pass
    except Exception:
        pass
    # builtin containers: sys.getsizeof (shallow)
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0


def snapshot(top=20, min_size=1024):
    """Collect a snapshot of live objects and print summaries.

    top: number of top items to show per category
    min_size: minimum bytes to include in 'big objects' listing
    """
    gc.collect()
    objects = gc.get_objects()
    type_sizes = defaultdict(int)
    type_counts = defaultdict(int)
    big_objects = []

    for o in objects:
        try:
            t = type(o)
            size = estimate_obj_size(o)
            type_sizes[t] += size
            type_counts[t] += 1
            if size >= min_size:
                big_objects.append((size, t, o))
        except Exception:
            continue

    # Report top types by total estimated bytes
    totals = [(s, t, type_counts[t]) for t, s in type_sizes.items()]
    totals.sort(reverse=True)

    print('\nTop types by estimated bytes:')
    for s, t, cnt in totals[:top]:
        print(f"{t.__module__}.{t.__name__}: {s} bytes across {cnt} objs")

    # Top big individual objects
    big_objects.sort(reverse=True, key=lambda x: x[0])
    print(f'\nTop {top} individual objects (>= {min_size} bytes):')
    for size, t, o in big_objects[:top]:
        try:
            preview = ''
            if isinstance(o, (str, bytes)):
                preview = repr(o)[:200]
            else:
                # try to extract useful attributes
                if hasattr(o, 'shape'):
                    preview = f'shape={getattr(o, "shape", None)}'
                elif hasattr(o, 'size') and not isinstance(o, (list, dict, tuple, str)):
                    preview = f'size={getattr(o, "size", None)}'
                else:
                    preview = repr(o)[:200]
            print(f"{size} bytes - {t.__module__}.{t.__name__} - {preview}")
        except Exception:
            print(f"{size} bytes - {t.__module__}.{t.__name__} - <repr failed>")

    # Also report top numpy arrays details if any
    if np is not None:
        np_objs = [(estimate_obj_size(o), o) for o in objects if isinstance(o, np.ndarray)]
        if np_objs:
            np_objs.sort(reverse=True, key=lambda x: x[0])
            print(f'\nTop {min(top, len(np_objs))} numpy arrays:')
            for size, arr in np_objs[:top]:
                try:
                    print(f'nbytes={size}, shape={getattr(arr, "shape", None)}, dtype={getattr(arr, "dtype", None)}')
                except Exception:
                    print(f'nbytes={size}, array_repr_failed')

    # PIL Images
    if Image is not None:
        pil_objs = [o for o in objects if isinstance(o, Image.Image)]
        if pil_objs:
            print(f'\nFound {len(pil_objs)} PIL.Image objects; listing top {min(top, len(pil_objs))}:')
            pil_sizes = []
            for o in pil_objs:
                try:
                    w, h = o.size
                    mode = o.mode
                    channels = 1 if mode == 'L' else 3
                    s = int(w * h * channels)
                    pil_sizes.append((s, o))
                except Exception:
                    pil_sizes.append((0, o))
            pil_sizes.sort(reverse=True, key=lambda x: x[0])
            for s, o in pil_sizes[:top]:
                print(f'size_est={s}, mode={getattr(o, "mode", None)}, size={getattr(o, "size", None)}')

    # Pygame surfaces
    if pygame is not None:
        surfaces = [o for o in objects if isinstance(o, pygame.Surface)]
        if surfaces:
            print(f'\nFound {len(surfaces)} pygame.Surface objects; listing top {min(top, len(surfaces))}:')
            surf_sizes = []
            for s in surfaces:
                try:
                    bpp = s.get_bytesize()
                    w, h = s.get_size()
                    est = int(w * h * bpp)
                except Exception:
                    est = 0
                surf_sizes.append((est, s))
            surf_sizes.sort(reverse=True, key=lambda x: x[0])
            for est, s in surf_sizes[:top]:
                try:
                    print(f'est={est}, size={s.get_size()}, flags={s.get_flags()}')
                except Exception:
                    print(f'est={est}, surface_repr_failed')

    # Optionally return a structured dict for programmatic use
    return {
        'type_totals': [(t.__module__ + '.' + t.__name__, s, type_counts[t]) for t, s in type_sizes.items()],
        'big_objects_count': len(big_objects)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--min', type=int, default=1024, help='minimum size (bytes) to include in big list')
    args = parser.parse_args()
    snapshot(top=args.top, min_size=args.min)


if __name__ == '__main__':
    main()
