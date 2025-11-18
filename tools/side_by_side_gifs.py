"""Create side-by-side GIFs/MP4s showing original vs contrastive trajectories.

Usage examples:
  python tools/side_by_side_gifs.py --traces results/run_*/Traces.pkl --out results/side_by_side
  python tools/side_by_side_gifs.py --traces traces/Traces.pkl --selected results/Selected_Highlights.pkl

The script looks for `State.image` or `State.img` for frames. If images are missing it will render a placeholder frame.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is on sys.path so pickled objects referencing
# local modules (e.g. counterfactual_outcomes) can be unpickled.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def to_image(x):
    # Accept numpy arrays or PIL Images; convert to uint8 RGB
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert('RGB')
    arr = np.array(x)
    if arr.dtype != np.uint8:
        try:
            arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype(np.uint8)
        except Exception:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr)


def placeholder(size, text):
    img = Image.new('RGB', size, (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w, h = draw.textsize(text, font=font)
    draw.text(((size[0]-w)//2, (size[1]-h)//2), text, fill=(200,200,200), font=font)
    return img


def compose_side_by_side(img1, img2, label1=None, label2=None):
    # make same height
    h = max(img1.height, img2.height)
    w1 = int(img1.width * (h / img1.height))
    w2 = int(img2.width * (h / img2.height))
    img1r = img1.resize((w1, h))
    img2r = img2.resize((w2, h))
    out = Image.new('RGB', (w1 + w2, h + 30), (0,0,0))
    out.paste(img1r, (0, 30))
    out.paste(img2r, (w1, 30))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if label1:
        draw.text((5, 5), label1, fill=(255,255,255), font=font)
    if label2:
        draw.text((w1 + 5, 5), label2, fill=(255,255,255), font=font)
    return out


def save_gif(frames, out_path, fps=3):
    duration = 1.0 / max(0.1, fps)
    imageio.mimsave(out_path, [np.array(f) for f in frames], duration=duration)


def save_mp4(frames, out_path, fps=3):
    writer = imageio.get_writer(out_path, fps=fps)
    for f in frames:
        writer.append_data(np.array(f))
    writer.close()


def process_traces(traces, out_dir, selected=None, fps=3, max_pairs=None):
    ensure_dir(out_dir)
    pairs_done = 0
    # selected can be a list of dicts with trace_idx and contrastive_idx or None
    if selected is not None:
        sel = selected
    else:
        # build list of (trace_idx, contrastive_idx)
        sel = []
        for t_idx, t in enumerate(traces):
            for c_idx, c in enumerate(getattr(t, 'contrastive', [])):
                sel.append({'trace_idx': t_idx, 'contrastive_idx': c_idx})
    for entry in sel:
        if max_pairs and pairs_done >= max_pairs:
            break
        # support multiple selected formats: dict {'trace_idx','contrastive_idx'},
        # tuple/list (trace_idx, contrastive_idx), or object with attributes
        if isinstance(entry, dict):
            t_idx = entry.get('trace_idx')
            c_idx = entry.get('contrastive_idx')
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            t_idx, c_idx = entry[0], entry[1]
        else:
            # try attribute-style (namedtuple) or fall back to skipping
            t_idx = getattr(entry, 'trace_idx', None)
            c_idx = getattr(entry, 'contrastive_idx', None)
        # handle nested tuple case like ((t_idx,c_idx), score) or similar
        # handle nested tuple case like ((t_idx,c_idx), score) or similar
        if isinstance(entry, (list, tuple)) and len(entry) >= 1 and isinstance(entry[0], (list, tuple)):
            # entry = ((t_idx, c_idx), <meta>)
            inner = entry[0]
            if len(inner) >= 2:
                t_idx, c_idx = inner[0], inner[1]
        elif (isinstance(t_idx, (list, tuple)) and c_idx is None) or (isinstance(t_idx, (list, tuple)) and isinstance(c_idx, (list, tuple))):
            if isinstance(t_idx, (list, tuple)) and len(t_idx) >= 2:
                t_idx, c_idx = t_idx[0], t_idx[1]
        if t_idx is None or c_idx is None:
            continue
        t = traces[t_idx]
        try:
            c = t.contrastive[c_idx]
        except Exception:
            continue
        # determine ranges
        # original frames: from c.start_idx to c.traj_end_state inclusive
        start = getattr(c, 'start_idx', 0)
        end = getattr(c, 'traj_end_state', len(t.states)-1)
        orig_frames = []
        contra_frames = []
        # Build aligned frame lists
        for i_rel, i in enumerate(range(start, end+1)):
            # original frame
            s = t.states[i]
            img = getattr(s, 'image', None) or getattr(s, 'img', None)
            img_pil = to_image(img) if img is not None else None
            if img_pil is None:
                img_pil = placeholder((200,200), f'orig t{t_idx} s{i}')
            orig_frames.append(img_pil)
            # contrastive frame: find matching state in c.states with same id[1]
            match = next((x for x in c.states if x.id[1] == i), None)
            if match is not None:
                img2 = getattr(match, 'image', None) or getattr(match, 'img', None)
                img2_pil = to_image(img2) if img2 is not None else None
                if img2_pil is None:
                    img2_pil = placeholder((200,200), f'contra t{t_idx} c{c_idx} s{i}')
            else:
                # if no matching state, use first contrastive frame or placeholder
                if c.states:
                    img2 = getattr(c.states[0], 'image', None) or getattr(c.states[0], 'img', None)
                    img2_pil = to_image(img2) if img2 is not None else placeholder((200,200), 'no_match')
                else:
                    img2_pil = placeholder((200,200), 'empty_contrastive')
            contra_frames.append(img2_pil)
        # Compose combined frames
        frames = []
        for o, cimg in zip(orig_frames, contra_frames):
            frames.append(compose_side_by_side(o, cimg, label1=f'orig t{t_idx}', label2=f'contra t{t_idx}-c{c_idx}'))
        base_name = f't{t_idx}_c{c_idx}'
        gif_path = os.path.join(out_dir, base_name + '.gif')
        mp4_path = os.path.join(out_dir, base_name + '.mp4')
        save_gif(frames, gif_path, fps=fps)
        try:
            save_mp4(frames, mp4_path, fps=fps)
        except Exception:
            pass
        pairs_done += 1
    return pairs_done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', help='Path to Traces.pkl', required=True)
    parser.add_argument('--selected', help='Path to Selected_Highlights.pkl (optional)')
    parser.add_argument('--out-dir', help='Output folder', default='results/side_by_side')
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--max', type=int, default=None)
    args = parser.parse_args()

    traces = load_pickle(args.traces)
    selected = None
    if args.selected:
        try:
            selected = load_pickle(args.selected)
        except Exception as e:
            print('Could not load selected highlights:', e)
            selected = None
    n = process_traces(traces, args.out_dir, selected=selected, fps=args.fps, max_pairs=args.max)
    print(f'Created {n} side-by-side animations in {args.out_dir}')

if __name__ == '__main__':
    main()
