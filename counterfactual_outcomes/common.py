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
    for hl in range(n_HLs):
        hl_str = str(hl) if hl > 9 else "0" + str(hl)
        img_array = []
        file_list = sorted(
            [x for x in glob.glob(frames_dir + "/*.png") if x.split('/')[-1].startswith(hl_str)])
        for i, f in enumerate(file_list):
            img = cv2.imread(f)
            if f.endswith("CA.png") and pause:
                [img_array.append(img) for _ in range(pause)]
            img_array.append(img)

        out = cv2.VideoWriter(join(video_dir, f'HL_{hl}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
    return len(img_array)


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


def append_trace_singlefile(path, trace):
    """Append a single pickled trace object to `path` (binary append).

    The file will contain multiple consecutive pickle objects and can be
    read back with `load_traces_multiobject`.
    """
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'ab') as f:
            pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # best-effort: if append fails, try atomic temp write then append
        try:
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.')
            os.close(fd)
            with open(tmp, 'wb') as t:
                pickle.dump(trace, t, protocol=pickle.HIGHEST_PROTOCOL)
            with open(tmp, 'rb') as t, open(path, 'ab') as f:
                f.write(t.read())
            try:
                os.remove(tmp)
            except Exception:
                pass
        except Exception:
            pass


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

def hstack_frames(img1, img2):
    if img1.shape[0] != img2.shape[0]:
        # resize to the smaller height
        min_height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * min_height / img1.shape[0]), min_height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * min_height / img2.shape[0]), min_height))
    
    combined_img = cv2.hconcat([img1, img2])
    return combined_img

def load_trace_from_file(file_path, trace_idx=None):
    """Load a single trace (converted to Trace object) from a Traces.pkl file.
    If trace_idx is provided, return the trace with matching `trace_idx` field.
    Otherwise return the first trace found.
    """
    if not os.path.exists(file_path):
        return None
    objs = []
    try:
        with open(file_path, 'rb') as f:
            while True:
                try:
                    objs.append(pickle.load(f))
                except EOFError:
                    break
    except Exception:
        return None

    # convert any dicts to Trace objects
    traces = [_dict_to_trace(o) for o in objs]
    if trace_idx is None:
        return traces[0] if traces else None
    for tr in traces:
        try:
            if getattr(tr, 'trace_idx', None) == trace_idx:
                return tr
        except Exception:
            continue
    return None

