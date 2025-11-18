"""Render side-by-side animations by re-rendering the Taxi environment for each saved State.

This script will:
- Load `Traces.pkl` and `Selected_Highlights.pkl` (optional).
- For each selected (trace_idx, contrastive_idx) pair, iterate through the trajectory frames.
- For each frame, set `env.unwrapped.s` to the integer stored in `State.state` and call `env.render()` to get an RGB frame.
- Compose original and contrastive frames side-by-side and save GIF/MP4.

Usage:
  python tools/render_side_by_side_env.py --traces results/.../Traces.pkl --selected results/.../Selected_Highlights.pkl --out-dir results/side_by_side_env --max 6

Note: This script specifically targets the Taxi environment (Taxi-v3). It assumes `State.state` stores the inner Taxi integer state.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# make repo root importable for pickles
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gym


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def to_image(arr):
    if isinstance(arr, Image.Image):
        return arr.convert('RGB')
    a = np.array(arr)
    if a.dtype != np.uint8:
        a = (255 * (a - a.min()) / (a.max() - a.min())).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[2] == 4:
        a = a[:, :, :3]
    return Image.fromarray(a)


def placeholder(size, text):
    img = Image.new('RGB', size, (40, 40, 40))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w, h = draw.textsize(text, font=font)
    draw.text(((size[0]-w)//2, (size[1]-h)//2), text, fill=(220,220,220), font=font)
    return img


def compose_side_by_side(img1, img2, label1=None, label2=None):
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


def render_state(env, state_obj):
    """Set the environment internal state and render an RGB frame.

    Expects `state_obj.state` to be an integer (Taxi internal state).
    """
    s = getattr(state_obj, 'state', None)
    try:
        s_int = int(s)
    except Exception:
        return None
    # create a fresh env instance to avoid side-effects
    # but to be faster we can reuse provided env
    try:
        # set on unwrapped where Taxi stores 's'
        env_unwrapped = env.unwrapped
        setattr(env_unwrapped, 's', s_int)
        # call render
        frame = env.render()
        # some gym versions require mode
        if frame is None:
            try:
                frame = env.render(mode='rgb_array')
            except Exception:
                frame = None
        return frame
    except Exception:
        return None


def custom_render_from_state(env, s_int, cell_size=64):
    """Stylized renderer: hedges, road, building, taxi, passenger and destinations.

    Falls back when `env.render()` is not available or fails.
    """
    try:
        un = env.unwrapped
        if not hasattr(un, 'decode'):
            return None
        taxi_row, taxi_col, pass_loc, dest_idx = un.decode(s_int)
        locs = getattr(un, 'locs', None)
        # create canvas (wider for nicer aspect)
        W = 5 * cell_size
        H = 5 * cell_size
        img = Image.new('RGB', (W, H), (120, 120, 120))
        draw = ImageDraw.Draw(img)

        # road background (grey textured)
        draw.rectangle((0, 0, W, H), fill=(115, 115, 115))

        # hedges positions (approximate, matching the attachment)
        hedge_color = (20, 140, 40)
        hedge_border = (10, 100, 30)
        hedges = [
            # top horizontal (columns 0,1) left, gap center, right (3,4)
            (0, 0, 2, 1),
            (3, 0, 5, 1),
            # bottom horizontal
            (0, 4, 2, 5),
            (3, 4, 5, 5),
            # two vertical hedges in middle
            (1, 1, 2, 4),
            (3, 1, 4, 4)
        ]
        for (c0, r0, c1, r1) in hedges:
            x0 = c0 * cell_size
            y0 = r0 * cell_size
            x1 = c1 * cell_size
            y1 = r1 * cell_size
            # rounded hedge
            draw.rounded_rectangle((x0+4, y0+4, x1-4, y1-4), radius=12, fill=hedge_color, outline=hedge_border)

        # draw building at bottom-left (cell 0,4)
        bx0 = 0 * cell_size + 8
        by0 = 4 * cell_size + 8
        bx1 = bx0 + cell_size - 16
        by1 = by0 + cell_size - 16
        draw.rectangle((bx0, by0, bx1, by1), fill=(200,180,120), outline=(120,80,40))
        # windows
        wx = bx0 + 6
        wy = by0 + 6
        for r in range(2):
            for c in range(2):
                draw.rectangle((wx + c*18, wy + r*18, wx + c*18 + 12, wy + r*18 + 12), fill=(180,220,255))

        # draw destination markers (red square with green background patch behind)
        if locs is not None:
            for idx, (lr, lc) in enumerate(locs):
                cx = lc * cell_size + cell_size//2
                cy = lr * cell_size + cell_size//2
                # draw green pad
                pad = 18
                draw.rectangle((cx-pad, cy-pad, cx+pad, cy+pad), fill=(30,200,80))
                # draw small building-like marker
                draw.rectangle((cx-10, cy-10, cx+10, cy+10), fill=(200,30,30))
                draw.text((cx-6, cy-8), str(idx), fill=(255,255,255))

        # draw passenger
        if pass_loc is not None and locs is not None:
            if pass_loc != 4:
                pr, pc = locs[pass_loc]
                cx = pc * cell_size + cell_size//2
                cy = pr * cell_size + cell_size//2
                # green carpet
                draw.rectangle((cx-14, cy-14, cx+14, cy+14), fill=(100,220,120))
                # passenger icon (circle + head)
                draw.ellipse((cx-8, cy-12, cx+8, cy+4), fill=(230,180,140))
                draw.ellipse((cx-5, cy-18, cx+5, cy-8), fill=(255,220,180))
            else:
                # passenger is in taxi; we'll draw indicator later
                pass

        # draw taxi (yellow car)
        tx = taxi_col * cell_size
        ty = taxi_row * cell_size
        car_pad = 8
        car_box = (tx+car_pad, ty+car_pad, tx+cell_size-car_pad, ty+cell_size-car_pad)
        draw.rounded_rectangle(car_box, radius=8, fill=(240,200,0), outline=(180,140,0))
        # windows
        wx0 = tx + car_pad + 6
        wy0 = ty + car_pad + 6
        draw.rectangle((wx0, wy0, wx0+12, wy0+8), fill=(40,60,120))
        # wheels
        wheel_r = 4
        draw.ellipse((tx+6, ty+cell_size-12, tx+6+wheel_r*2, ty+cell_size-12+wheel_r*2), fill=(30,30,30))
        draw.ellipse((tx+cell_size-14, ty+cell_size-12, tx+cell_size-14+wheel_r*2, ty+cell_size-12+wheel_r*2), fill=(30,30,30))

        # passenger-in-taxi indicator
        if pass_loc == 4:
            cx = taxi_col * cell_size + cell_size//2
            cy = taxi_row * cell_size + cell_size//2
            draw.ellipse((cx-8, cy-8, cx+8, cy+8), fill=(40,160,40))

        return np.array(img)
    except Exception:
        return None

def process(traces, selected, out_dir, fps=3, max_pairs=6):
    ensure_dir(out_dir)
    env = None
    try:
        env = gym.make('Taxi-v3')
    except Exception:
        try:
            env = gym.make('Taxi-v3', render_mode='rgb_array')
        except Exception:
            env = None

    pairs = []
    if selected:
        # selected may be list of ((t_idx,c_idx), score) or list of (t_idx,c_idx)
        for item in selected:
            if isinstance(item, tuple) and len(item) >= 1:
                k = item[0]
                if isinstance(k, tuple) and len(k) >= 2:
                    pairs.append((int(k[0]), int(k[1])))
                elif isinstance(item, tuple) and len(item) >= 2:
                    pairs.append((int(item[0]), int(item[1])))
    else:
        # fallback: pair first N traces with themselves
        for i in range(min(max_pairs, len(traces))):
            pairs.append((i, i))

    count = 0
    for idx, (ti, ci) in enumerate(pairs[:max_pairs]):
        try:
            t = traces[ti]
            # handle contrastive index out-of-range: map into available traces
            if ci < 0:
                raise IndexError('negative index')
            if ci >= len(traces):
                mapped = int(ci) % len(traces)
                print(f'contrastive index {ci} out of range, mapping to {mapped}')
                c = traces[mapped]
            else:
                c = traces[ci]
        except Exception:
            continue

        # access list of states
        st_a = getattr(t, 'states', None) or getattr(t, 'trajectory', None) or t
        st_b = getattr(c, 'states', None) or getattr(c, 'trajectory', None) or c
        # ensure iterable
        try:
            len_a = len(st_a)
        except Exception:
            st_a = list(st_a)
        try:
            len_b = len(st_b)
        except Exception:
            st_b = list(st_b)

        L = min(len(st_a), len(st_b))
        # limit frames to keep GIFs manageable
        max_frames = 60
        if L <= max_frames:
            indices = list(range(L))
        else:
            step = max(1, L // max_frames)
            indices = list(range(0, L, step))
            if indices[-1] != L - 1:
                indices.append(L - 1)

        frames = []
        for i in indices:
            sa = st_a[i]
            sb = st_b[i]
            # get images
            img_a = None
            img_b = None
            if hasattr(sa, 'image') or hasattr(sa, 'img'):
                img_a = getattr(sa, 'image', getattr(sa, 'img', None))
                try:
                    img_a = to_image(img_a)
                except Exception:
                    img_a = None
            if hasattr(sb, 'image') or hasattr(sb, 'img'):
                img_b = getattr(sb, 'image', getattr(sb, 'img', None))
                try:
                    img_b = to_image(img_b)
                except Exception:
                    img_b = None

            if img_a is None and env is not None:
                f = render_state(env, sa)
                if f is not None:
                    img_a = to_image(f)
            if img_a is None and env is not None:
                try:
                    s_int = int(getattr(sa, 'state', None))
                    f = custom_render_from_state(env, s_int)
                    if f is not None:
                        img_a = Image.fromarray(f)
                except Exception:
                    img_a = None

            if img_b is None and env is not None:
                f = render_state(env, sb)
                if f is not None:
                    img_b = to_image(f)
            if img_b is None and env is not None:
                try:
                    s_int = int(getattr(sb, 'state', None))
                    f = custom_render_from_state(env, s_int)
                    if f is not None:
                        img_b = Image.fromarray(f)
                except Exception:
                    img_b = None

            if img_a is None:
                img_a = placeholder((320, 320), 'no frame A')
            if img_b is None:
                img_b = placeholder((320, 320), 'no frame B')

            frames.append(compose_side_by_side(img_a, img_b, label1=f't{ti}', label2=f'c{ci}'))

        if frames:
            outp = os.path.join(out_dir, f't{ti}_c{ci}.gif')
            save_gif(frames, outp, fps=fps)
            count += 1

    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', required=True)
    parser.add_argument('--selected', required=False)
    parser.add_argument('--out-dir', default='results/side_by_side_env')
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--max', type=int, default=6)
    args = parser.parse_args()

    traces = load_pickle(args.traces)
    selected = None
    if args.selected:
        try:
            selected = load_pickle(args.selected)
        except Exception as e:
            print('could not load selected highlights', e)
            selected = None
    n = process(traces, selected, args.out_dir, fps=args.fps, max_pairs=args.max)
    print('created', n, 'rendered side-by-side animations in', args.out_dir)


if __name__ == '__main__':
    main()
