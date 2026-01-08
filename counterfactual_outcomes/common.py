import logging

import os
import glob
import pickle
from os.path import join

import cv2
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import numpy as np
import shutil
import os

class Trace(object):
    def __init__(self, idx, k_steps):
        self.obs = []
        self.previous_actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.reward_sum = 0
        self.length = 0
        self.states = []
        # list of ContrastiveTrajectory objects (filled by contrastive_online)
        self.contrastive = []
        self.trace_idx = idx
        self.k_steps = k_steps

    def update(self, obs, r, done, infos, a, state_id):
        self.obs.append(obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.previous_actions.append(a)
        self.reward_sum += r
        self.states.append(state_id)
        self.length += 1

    def get_traj_frames(self, idxs):
        frames = []
        for i in idxs:
            frames.append(self.states[i].image)
        return frames

    def mark_frames(self, hl_idx, indexes, color=255, thickness=2, no_mark=False):
        """Generic mark_frames for base Trace: returns list of frames (arrays or paths)
        and the relative index of the highlighted state. Works with State objects
        or compact dicts produced when traces are serialized.
        """
        frames = []
        rel_idx = 0

        if not indexes:
            return frames, rel_idx

        start = indexes[0]
        end = indexes[-1]
        # clamp to available states
        start = max(0, start)
        end = min(len(self.states) - 1, end) if self.states else end

        for i in range(start, end + 1):
            st = self.states[i]
            img = None
            # st may be a State object or a compact dict
            if isinstance(st, dict):
                # dict may contain 'image_path' or 'image'
                img = st.get('image', None)
                if img is None:
                    img = st.get('img', None)
                if img is None:
                    img = st.get('image_path', None)
            else:
                img = getattr(st, 'image', None) if hasattr(st, 'image') else None
                if img is None:
                    img = getattr(st, 'img', None) if hasattr(st, 'img') else None
                if img is None:
                    img = getattr(st, 'image_path', None) if hasattr(st, 'image_path') else None

            if img is None:
                # placeholder tiny white image path (None will be handled later by save_frames)
                frames.append(None)
            else:
                frames.append(img)

        # compute relative index
        if hl_idx < start or hl_idx > end:
            rel_idx = 0
        else:
            rel_idx = hl_idx - start

        return frames, rel_idx


class State(object):
    def __init__(self, id, obs, state, action_vector, img, features):
        self.id = id
        self.obs = obs
        self.state = state
        self.action_vector = action_vector
        # ensure both attribute names exist
        self.img = img
        self.image = img
        self.features = features

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


def get_highlight_traj_indxs(highlights):
    traj_indxs = {}
    for hl in highlights:
        traj_indxs[(hl.id[0], hl.id[1])] = [x.id[1] for x in hl.states if
                                            x.id[1] <= hl.traj_end_state]
    return traj_indxs


def save_frames(trajectories_dict, path, contra_rel_idxs=False):
    make_clean_dirs(path)
    for i, hl in enumerate(trajectories_dict):
        for j, f in enumerate(trajectories_dict[hl]):
            vid_num = str(i) if i > 9 else "0" + str(i)
            frame_num = str(j) if j > 9 else "0" + str(j)
            img_name = f"{vid_num}_{frame_num}"
            if contra_rel_idxs and  j == contra_rel_idxs[hl]:
                img_name += "_CA"
            save_image(path, img_name, f)

def save_highlights(img_shape, n_videos, frames_path, videos_path, args):
    """Save Highlight videos"""
    height, width, layers = img_shape
    img_size = (width, height)
    create_highlights_videos(frames_path, videos_path, n_videos, img_size,
                             args.fps, pause=args.pause)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def pickle_save(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_traces(path):
    # Support both old-style single-list pickle and new-style appended-multiple-objects
    p = join(path, 'Traces.pkl')
    if os.path.exists(p):
        try:
            objs = load_traces_multiobject(p)
            # convert any plain-dict traces (compact format) back into Trace/State objects
            return [_dict_to_trace(o) for o in objs]
        except Exception:
            try:
                objs = pickle_load(p)
                # if a single-list was stored, convert dict entries as well
                if isinstance(objs, list):
                    return [_dict_to_trace(o) for o in objs]
                return _dict_to_trace(objs)
            except Exception:
                return []
    return []


def save_traces(traces, output_dir, name='Traces.pkl'):
    try:
        os.makedirs(output_dir)
    except:
        pass
    pickle_save(traces, join(output_dir, name))


def make_clean_dirs(path, no_clean=False, file_type=''):
    try:
        os.makedirs(path)
    except:
        if not no_clean: clean_dir(path, file_type)


def clean_dir(path, file_type=''):
    files = glob.glob(path + "/*" + file_type)
    for f in files:
        os.remove(f)


def create_highlights_videos(frames_dir, video_dir, n_HLs, size, fps, pause=None):
    make_clean_dirs(video_dir)
    total_frames = 0
    for hl in range(n_HLs):
        hl_str = str(hl) if hl > 9 else "0" + str(hl)
        pattern = os.path.join(frames_dir, "*.png")
        file_list = sorted([x for x in glob.glob(pattern) if os.path.basename(x).startswith(hl_str)])

        # If there are no frames for this highlight, skip
        if not file_list:
            continue

        # Determine common size from the first frame
        first_frame = cv2.imread(file_list[0])
        common_size = (first_frame.shape[1], first_frame.shape[0])

        img_array = []
        for i, f in enumerate(file_list):
            img = cv2.imread(f)
            # Resize the frame to the common size
            img = cv2.resize(img, common_size)
            if f.endswith("CA.png") and pause:
                [img_array.append(img) for _ in range(pause)]
            img_array.append(img)

        out = cv2.VideoWriter(join(video_dir, f'HL_{hl}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, common_size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        total_frames = len(img_array)
    return total_frames


def save_image(path, name, img):
    # img may be a numpy array or other image-like object; ensure uint8
    out_path = path + '/' + name + '.png'
    try:
        imageio.imsave(out_path, img_as_ubyte(img))
    except Exception:
        try:
            # If img is already a path, copy it
            if isinstance(img, str) and os.path.exists(img):
                shutil.copy(img, out_path)
                return
        except Exception:
            pass
        # final fallback: write a tiny white image
        imageio.imsave(out_path, (255 * np.ones((1, 1, 3), dtype='uint8')))


def log_msg(msg, verbose=True):

    logging.info(msg)


# def append_trace_singlefile(path, trace):
#     """Append a single pickled trace object to `path` (binary append).

#     The file will contain multiple consecutive pickle objects and can be
#     read back with `load_traces_multiobject`.
#     """
#     try:
#         d = os.path.dirname(path)
#         if d:
#             os.makedirs(d, exist_ok=True)
#         with open(path, 'ab') as f:
#             pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)
#     except Exception:
#         # best-effort: if append fails, try atomic temp write then append
#         try:
#             import tempfile
#             fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.')
#             os.close(fd)
#             with open(tmp, 'wb') as t:
#                 pickle.dump(trace, t, protocol=pickle.HIGHEST_PROTOCOL)
#             with open(tmp, 'rb') as t, open(path, 'ab') as f:
#                 f.write(t.read())
#             try:
#                 os.remove(tmp)
#             except Exception:
#                 pass
#         except Exception:
#             pass


def load_traces_multiobject(path):
    """Load multiple pickled objects from a single file written by
    `append_trace_singlefile`. Returns a list of loaded objects.
    """
    objs = []
    if not os.path.exists(path):
        return objs
    try:
        with open(path, 'rb') as f:
            while True:
                try:
                    objs.append(pickle.load(f))
                except EOFError:
                    break
    except Exception:
        # In case of a partial/corrupt file, try to read until failure
        try:
            with open(path, 'rb') as f:
                while True:
                    objs.append(pickle.load(f))
        except Exception:
            pass
    return objs


def _dict_to_trace(o):
    """Convert a compact trace dictionary into a Trace-like object (Trace + State instances).
    If the object is already a Trace (or not a dict), return it unchanged.
    """
    from types import SimpleNamespace

    if not isinstance(o, dict):
        return o

    # if it doesn't look like a trace dict, return as-is
    if 'states' not in o:
        return o

    # Create a Trace instance and populate fields from dict
    try:
        tr = Trace(o.get('trace_idx', 0), o.get('k_steps', o.get('k_steps', 0)))
    except Exception:
        tr = Trace(o.get('trace_idx', 0), 0)

    # basic lists
    tr.previous_actions = o.get('previous_actions', []) or []
    tr.rewards = o.get('rewards', []) or []
    tr.dones = o.get('dones', []) or []
    tr.infos = o.get('infos', []) or []
    tr.reward_sum = o.get('reward_sum', 0)
    tr.length = o.get('length', 0)

    # convert states: dict -> State
    tr.states = []
    for s in o.get('states', []) or []:
        # s is expected to be a dict with keys similar to State
        st = State(s.get('id'), s.get('obs', None), s.get('state', None), s.get('action_vector', None), None, s.get('features', None))
        # preserve image_path if available, set image to None to avoid heavy in-memory arrays
        if isinstance(s, dict) and s.get('image_path'):
            setattr(st, 'image_path', s.get('image_path'))
            st.image = None
            st.img = None
        else:
            # if raw image present, keep it (best-effort)
            st.image = s.get('image', None) if isinstance(s, dict) else None
            st.img = st.image
        tr.states.append(st)

    # convert contrastive entries (if present)
    tr.contrastive = []
    for c in o.get('contrastive', []) or []:
        cobj = SimpleNamespace()
        # shallow copy simple attributes (except nested states)
        for k, v in (c.items() if isinstance(c, dict) else []):
            if k == 'states':
                continue
            setattr(cobj, k, v)
        # ensure common attributes exist with safe defaults
        if not hasattr(cobj, 'actions'):
            setattr(cobj, 'actions', c.get('actions', []) if isinstance(c, dict) else [])
        if not hasattr(cobj, 'rewards'):
            setattr(cobj, 'rewards', c.get('rewards', []) if isinstance(c, dict) else [])
        if not hasattr(cobj, 'importance'):
            setattr(cobj, 'importance', c.get('importance', 0) if isinstance(c, dict) else 0)
        if not hasattr(cobj, 'start_idx'):
            setattr(cobj, 'start_idx', c.get('start_idx', None) if isinstance(c, dict) else None)
        # convert states inside contrastive
        c_states = []
        for s in (c.get('states', []) if isinstance(c, dict) else []):
            st = State(s.get('id'), s.get('obs', None), s.get('state', None), s.get('action_vector', None), None, s.get('features', None))
            if isinstance(s, dict) and s.get('image_path'):
                setattr(st, 'image_path', s.get('image_path'))
                st.image = None
                st.img = None
            else:
                st.image = s.get('image', None) if isinstance(s, dict) else None
                st.img = st.image
            c_states.append(st)
        setattr(cobj, 'states', c_states)
        # ensure typical attributes exist
        if not hasattr(cobj, 'id'):
            setattr(cobj, 'id', getattr(cobj, 'trace_idx', (None, None)))
        if not hasattr(cobj, 'start_idx'):
            setattr(cobj, 'start_idx', getattr(cobj, 'id', (None, None))[1] if getattr(cobj, 'id', None) else 0)
        tr.contrastive.append(cobj)

    # preserve RD_vals if present
    if 'RD_vals' in o:
        tr.RD_vals = o.get('RD_vals')

    return tr


def iter_load_traces_multiobject(path):
    """Yield pickled objects from a file written by `append_trace_singlefile`.
    This allows streaming consumption without building the full list in memory.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    except Exception:
        return

def hstack_frames(img1, text1, img2, text2):
    
    # --- Step 1: Define Constants and Calculate Required Caption Heights ---
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    padding = 20                 # Vertical padding around text
    text_color = (0, 0, 0)       # Black text (BGR)
    bg_color = (255, 255, 255)   # White background (BGR)

    # Calculate required height for text 1
    (text_w1, text_h1), baseline1 = cv2.getTextSize(text1, font, font_scale, font_thickness)
    caption_height1 = text_h1 + baseline1 + (padding * 2)
    
    # Calculate required height for text 2
    (text_w2, text_h2), baseline2 = cv2.getTextSize(text2, font, font_scale, font_thickness)
    caption_height2 = text_h2 + baseline2 + (padding * 2)
    
    # Determine the maximum height to ensure clean concatenation
    max_caption_height = max(caption_height1, caption_height2)

    # --- Step 2: Handle Original Image Height Mismatch ---
    
    if img1.shape[0] != img2.shape[0]:
        # Resize to the smaller height, maintaining aspect ratio
        min_original_height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * min_original_height / img1.shape[0]), min_original_height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * min_original_height / img2.shape[0]), min_original_height))

    # --- Step 3: Process and Caption Images (Using a local function for clean repetition) ---
    
    def caption_and_pad(img, text, text_w, text_h, baseline):
        img_h, img_w = img.shape[:2]
        
        # 3a. Create the border (padding the bottom to max_caption_height)
        img_with_border = cv2.copyMakeBorder(
            img, 
            0,                     # top
            max_caption_height,    # bottom (uses the calculated max height)
            0, 
            0, 
            cv2.BORDER_CONSTANT, 
            value=bg_color
        )
        
        # 3b. Calculate coordinates to center text in the new white bar
        
        # X: Center text horizontally
        text_x = (img_w - text_w) // 2
        
        # Y: Center text vertically within the white bar
        # Start at original image height (img_h) + half the bar height + half the text height - baseline offset
        text_y = img_h + (max_caption_height // 2) + (text_h // 2) - baseline
        
        # 3c. Draw the text
        cv2.putText(
            img_with_border, 
            text, 
            (text_x, text_y), 
            font, 
            font_scale, 
            text_color, 
            font_thickness, 
            cv2.LINE_AA
        )
        return img_with_border

    # Process both images using the local function
    final_img1 = caption_and_pad(img1, text1, text_w1, text_h1, baseline1)
    final_img2 = caption_and_pad(img2, text2, text_w2, text_h2, baseline2)

    # --- Step 4: Horizontal Concatenation ---
    
    # Since final_img1 and final_img2 have the same total height, hconcat works perfectly.
    combined_img = cv2.hconcat([final_img1, final_img2])
    
    return combined_img

def mark_right_half_counterfactual(frame_bgr, is_counterfactual=False, color=(0, 0, 255), thickness=6, tint_alpha=0.12):
    """Mark the right half of a combined (left|right) frame as counterfactual.
    - If `is_counterfactual` True, draws a border on the right half and applies a slight tint.
    - color is BGR tuple.
    """
    import numpy as _np
    import cv2 as _cv

    out = frame_bgr.copy()
    h, w = out.shape[:2]
    half = w // 2
    if not is_counterfactual:
        return out
    # draw border around right half
    _cv.rectangle(out, (half, 0), (w - 1, h - 1), color, thickness)
    # apply tint
    tint = _np.full((h, w - half, 3), color, dtype='uint8')
    alpha = float(tint_alpha)
    right = out[:, half:w].astype('float32')
    blended = (1 - alpha) * right + alpha * tint.astype('float32')
    out[:, half:w] = blended.astype('uint8')
    return out


def create_reward_bar_chart(frame_width, left_rewards, right_rewards, current_step_idx,
                            hist_h=100, metadata_lines=None):
    """
    Create a BGR bar chart image of height `hist_h` + metadata space.
    Draws two side-by-side charts for left_rewards and right_rewards.
    - Bars are red (<0) or green (>=0).
    - A specific marker/highlight is drawn at `current_step_idx`.
    """
    import numpy as _np
    import cv2 as _cv

    W = int(frame_width)
    padding = 10
    
    # --- Metadata drawing ---
    meta_h = 0
    font = _cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    
    if metadata_lines:
        lines = metadata_lines if isinstance(metadata_lines, list) else str(metadata_lines).split('\n')
        line_spacing = 20
        meta_h = len(lines) * line_spacing + padding * 2
    
    left_margin = 60
    idx_h = 45 # Increased for X-axis labels
    total_h = hist_h + meta_h + idx_h
    
    # Create white canvas
    canvas = _np.full((total_h, W, 3), 255, dtype=_np.uint8)

    # Draw Metadata
    if metadata_lines:
        y_text = padding + 12
        for ln in lines:
            text_size, _ = _cv.getTextSize(ln, font, font_scale, thickness)
            tx = (W - text_size[0]) // 2
            _cv.putText(canvas, ln, (tx, y_text), font, font_scale, (50, 50, 50), thickness, _cv.LINE_AA)
            y_text += line_spacing

    # --- Draw Legend ---
    legend_y = padding + 12
    legend_x = W - 110
    # Positive
    _cv.rectangle(canvas, (legend_x, legend_y - 8), (legend_x + 10, legend_y + 2), (100, 200, 100), -1)
    _cv.putText(canvas, "Positive", (legend_x + 15, legend_y), font, 0.35, (50, 50, 50), 1, _cv.LINE_AA)
    # Negative
    _cv.rectangle(canvas, (legend_x, legend_y + 12), (legend_x + 10, legend_y + 22), (100, 100, 220), -1)
    _cv.putText(canvas, "Negative", (legend_x + 15, legend_y + 20), font, 0.35, (50, 50, 50), 1, _cv.LINE_AA)

    # --- Setup Chart Areas ---
    chart_y_start = meta_h + padding + 15 # extra space for sub-titles
    chart_h = hist_h
    
    # Determine Global Min/Max
    valid_l = [r for r in left_rewards if r is not None and _np.isfinite(r)]
    valid_r = [r for r in right_rewards if r is not None and _np.isfinite(r)]
    all_vals = valid_l + valid_r
    
    if not all_vals:
        max_val = 1.0
        min_val = -1.0
    else:
        max_val = max(1.0, max(all_vals))
        min_val = min(-1.0, min(all_vals))
        
    max_val *= 1.1
    min_val *= 1.1
    val_range = max_val - min_val
    if val_range == 0: val_range = 1.0

    # Draw Y-Axis Label (Vertical)
    y_label_text = "Reward (r)"
    # Rotate text by drawing onto a separate surface or just character by character? 
    # Simple way: character by character or just horizontal on the side.
    # We'll do a simple vertical stack for "Reward (r)"
    y_lab_start = chart_y_start + (chart_h // 2) - 30
    for i, char in enumerate(y_label_text):
        _cv.putText(canvas, char, (10, y_lab_start + i*12), font, 0.35, (0, 0, 0), 1, _cv.LINE_AA)

    # Draw Min/Max scale values
    _cv.putText(canvas, f"{max_val:.1f}", (left_margin - 35, chart_y_start + 10), font, 0.3, (100, 100, 100), 1, _cv.LINE_AA)
    _cv.putText(canvas, f"{min_val:.1f}", (left_margin - 35, chart_y_start + chart_h), font, 0.3, (100, 100, 100), 1, _cv.LINE_AA)

    def _draw_chart(rewards, x_offset, width, title):
        # Draw Title
        (tw, th), _ = _cv.getTextSize(title, font, 0.4, 1)
        _cv.putText(canvas, title, (x_offset + (width-tw)//2, chart_y_start - 10), font, 0.4, (0,0,0), 1, _cv.LINE_AA)

        # Draw zero line
        zero_ratio = (0 - min_val) / val_range
        zero_y = int(chart_y_start + chart_h - (zero_ratio * chart_h))
        _cv.line(canvas, (x_offset, zero_y), (x_offset + width, zero_y), (200, 200, 200), 1)
        
        n_steps = len(rewards)
        if n_steps == 0: return

        bar_w = width / n_steps
        curr_x_center = -1

        for i, r in enumerate(rewards):
            x1 = int(x_offset + i * bar_w)
            x2 = int(x_offset + (i + 1) * bar_w) - 1
            if x2 < x1: x2 = x1
            if i == current_step_idx:
                curr_x_center = (x1 + x2) // 2
            if r is None or not _np.isfinite(r):
                continue
            
            if r >= 0:
                top_val, bot_val, color = r, 0, (100, 200, 100)
            else:
                top_val, bot_val, color = 0, r, (100, 100, 220)
            
            top_y_ratio = (top_val - min_val) / val_range
            bot_y_ratio = (bot_val - min_val) / val_range
            y1 = int(chart_y_start + chart_h - (top_y_ratio * chart_h))
            y2 = int(chart_y_start + chart_h - (bot_y_ratio * chart_h))
            _cv.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            
        if curr_x_center >= 0:
             _cv.line(canvas, (curr_x_center, chart_y_start), (curr_x_center, chart_y_start + chart_h), (50, 50, 50), 1)

        # Draw X-axis label "Step"
        (sw, sh), _ = _cv.getTextSize("Step", font, 0.4, 1)
        _cv.putText(canvas, "Step", (x_offset + (width-sw)//2, chart_y_start + chart_h + 20), font, 0.4, (0,0,0), 1, _cv.LINE_AA)
        _cv.putText(canvas, "0", (x_offset, chart_y_start + chart_h + 12), font, 0.3, (100, 100, 100), 1, _cv.LINE_AA)
        _cv.putText(canvas, str(n_steps), (x_offset + width - 15, chart_y_start + chart_h + 12), font, 0.3, (100, 100, 100), 1, _cv.LINE_AA)

    chart_w = (W - left_margin - 30) // 2
    _draw_chart(left_rewards, left_margin, chart_w, "ORIGINAL")
    _draw_chart(right_rewards, left_margin + chart_w + 20, chart_w, "COUNTERFACTUAL")
    
    # Draw Separator
    _cv.line(canvas, (left_margin + chart_w + 10, chart_y_start - 20), (left_margin + chart_w + 10, total_h - 10), (220, 220, 220), 1)
    
    return canvas
    
    return canvas

def create_info_strip(width, text_left, text_right, height=40, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    """Creates a horizontal strip with two centered text labels (left and right halves)."""
    strip = np.full((height, width, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    
    # Left text
    (tw, th), baseline = cv2.getTextSize(text_left, font, scale, thickness)
    x = (width // 4) - (tw // 2)
    y = (height // 2) + (th // 2)
    cv2.putText(strip, text_left, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
    
    # Right text
    (tw, th), baseline = cv2.getTextSize(text_right, font, scale, thickness)
    x = (3 * (width // 4)) - (tw // 2)
    y = (height // 2) + (th // 2)
    cv2.putText(strip, text_right, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
    
    return strip




