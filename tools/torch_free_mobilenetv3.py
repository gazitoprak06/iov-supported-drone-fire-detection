"""Torch-free loader + forward pass for the trained MobileNetV3-Small fire detector.

Why this exists
---------------
The evaluation must be reproducible on a machine that has no PyTorch install
(e.g. a CI container, or a reviewer's box). This module:

  1. reads `v3_mobilenet.pth` (a torch zip archive) into plain NumPy arrays
     without importing torch, and
  2. re-implements the torchvision `mobilenet_v3_small` forward pass in NumPy.

Validation status
-----------------
`tools/verify_numpy_model.py` compares this forward pass against torchvision's
on real frames, but it requires PyTorch and so cannot run in a torch-free
environment. Until it has been run and has passed on a machine with PyTorch,
treat any metric produced through this module as provisional: the structural
checks below (weight shapes, a sane validation-set accuracy) rule out a grossly
wrong implementation, but they do not establish bit-level agreement, and clip
verdicts are argmax decisions that a small logit error can flip.
"""

import io
import pickle
import zipfile

import numpy as np

# --------------------------------------------------------------------------
# 1. Read a torch .pth (zip serialisation) into numpy arrays, without torch
# --------------------------------------------------------------------------

_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _FakeStorage:
    def __init__(self, dtype, key, numel):
        self.dtype = dtype
        self.key = key
        self.numel = numel


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_):
    arr = storage["array"]
    size = tuple(size)
    if not size:
        return arr[storage_offset]
    count = int(np.prod(size))
    flat = arr[storage_offset:storage_offset + count]
    return flat.reshape(size)


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, zf):
        super().__init__(file, encoding="utf-8")
        self._zf = zf

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in _DTYPES:
            return _DTYPES[name]
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # pid = ('storage', storage_type, key, location, numel)
        _, storage_type, key, _location, numel = pid
        dtype = storage_type if isinstance(storage_type, type) else _DTYPES[storage_type]
        name = None
        for candidate in self._zf.namelist():
            if candidate.endswith(f"data/{key}"):
                name = candidate
                break
        if name is None:
            raise KeyError(f"storage {key} not found in archive")
        raw = self._zf.read(name)
        return {"array": np.frombuffer(raw, dtype=dtype).copy()}


def load_state_dict(path):
    """Return {param_name: np.ndarray} from a torch-saved state_dict."""
    with zipfile.ZipFile(path) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        with zf.open(pkl_name) as fh:
            data = _Unpickler(io.BytesIO(fh.read()), zf).load()
    return {k: np.asarray(v) for k, v in data.items()}


# --------------------------------------------------------------------------
# 2. NumPy layers
# --------------------------------------------------------------------------

def conv2d(x, w, b=None, stride=1, padding=0, groups=1):
    """x: (C,H,W) float32.  w: (out, in/groups, kh, kw).  Returns (out,H',W')."""
    c_in, h, w_in = x.shape
    out_c, in_per_g, kh, kw = w.shape
    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    h_pad, w_pad = x.shape[1], x.shape[2]
    h_out = (h_pad - kh) // stride + 1
    w_out = (w_pad - kw) // stride + 1

    s = x.strides
    # sliding windows: (C, h_out, w_out, kh, kw)
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(c_in, h_out, w_out, kh, kw),
        strides=(s[0], s[1] * stride, s[2] * stride, s[1], s[2]),
    )

    if groups == 1:
        cols = windows.transpose(1, 2, 0, 3, 4).reshape(h_out * w_out, c_in * kh * kw)
        out = cols @ w.reshape(out_c, -1).T                       # (HW, out_c)
        out = out.T.reshape(out_c, h_out, w_out)
    elif groups == c_in and in_per_g == 1:
        # depthwise: per-channel dot product
        cols = windows.reshape(c_in, h_out * w_out, kh * kw)      # (C, HW, K)
        ker = w.reshape(c_in, kh * kw)                            # (C, K)
        out = np.einsum("chk,ck->ch", cols, ker).reshape(c_in, h_out, w_out)
    else:
        outs = []
        c_per_g = c_in // groups
        o_per_g = out_c // groups
        for g in range(groups):
            cg = windows[g * c_per_g:(g + 1) * c_per_g]
            cols = cg.transpose(1, 2, 0, 3, 4).reshape(h_out * w_out, c_per_g * kh * kw)
            wg = w[g * o_per_g:(g + 1) * o_per_g].reshape(o_per_g, -1)
            outs.append((cols @ wg.T).T.reshape(o_per_g, h_out, w_out))
        out = np.concatenate(outs, axis=0)

    if b is not None:
        out += b[:, None, None]
    return np.ascontiguousarray(out, dtype=np.float32)


def batchnorm(x, gamma, beta, mean, var, eps=1e-5):
    scale = gamma / np.sqrt(var + eps)
    return x * scale[:, None, None] + (beta - mean * scale)[:, None, None]


def relu(x):
    return np.maximum(x, 0.0)


def hardswish(x):
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def hardsigmoid(x):
    return np.clip(x + 3.0, 0.0, 6.0) / 6.0


# --------------------------------------------------------------------------
# 3. torchvision mobilenet_v3_small topology
#    (in_c, kernel, expanded_c, out_c, use_se, activation, stride)
# --------------------------------------------------------------------------

BNECK_CONFIG = [
    (16, 3, 16, 16, True, "RE", 2),
    (16, 3, 72, 24, False, "RE", 2),
    (24, 3, 88, 24, False, "RE", 1),
    (24, 5, 96, 40, True, "HS", 2),
    (40, 5, 240, 40, True, "HS", 1),
    (40, 5, 240, 40, True, "HS", 1),
    (40, 5, 120, 48, True, "HS", 1),
    (48, 5, 144, 48, True, "HS", 1),
    (48, 5, 288, 96, True, "HS", 2),
    (96, 5, 576, 96, True, "HS", 1),
    (96, 5, 576, 96, True, "HS", 1),
]

ACT = {"RE": relu, "HS": hardswish}


class MobileNetV3SmallNumpy:
    def __init__(self, state_dict):
        self.sd = state_dict
        self._validate()

    # -- helpers ----------------------------------------------------------
    def _cbn(self, x, prefix, stride, padding, groups, act):
        w = self.sd[f"{prefix}.0.weight"]
        x = conv2d(x, w, None, stride=stride, padding=padding, groups=groups)
        x = batchnorm(
            x,
            self.sd[f"{prefix}.1.weight"], self.sd[f"{prefix}.1.bias"],
            self.sd[f"{prefix}.1.running_mean"], self.sd[f"{prefix}.1.running_var"],
        )
        return act(x) if act is not None else x

    def _se(self, x, prefix):
        scale = x.mean(axis=(1, 2), keepdims=True)                    # (C,1,1)
        s = conv2d(scale, self.sd[f"{prefix}.fc1.weight"], self.sd[f"{prefix}.fc1.bias"])
        s = relu(s)
        s = conv2d(s, self.sd[f"{prefix}.fc2.weight"], self.sd[f"{prefix}.fc2.bias"])
        return x * hardsigmoid(s)

    def _validate(self):
        """Cheap structural check: every weight the forward pass will ask for exists."""
        required = ["features.0.0.weight", "features.12.0.weight",
                    "classifier.0.weight", "classifier.3.weight"]
        missing = [k for k in required if k not in self.sd]
        if missing:
            raise KeyError(f"state_dict is missing {missing}")
        n_classes = self.sd["classifier.3.weight"].shape[0]
        if n_classes != 2:
            raise ValueError(f"expected a 2-class head, got {n_classes}")

    # -- forward ----------------------------------------------------------
    def forward(self, x):
        """x: (3,224,224) already normalised. Returns logits (2,)."""
        x = self._cbn(x, "features.0", stride=2, padding=1, groups=1, act=hardswish)

        for i, (in_c, k, exp_c, out_c, use_se, nl, stride) in enumerate(BNECK_CONFIG, start=1):
            act = ACT[nl]
            identity = x
            sub = 0
            if exp_c != in_c:
                x = self._cbn(x, f"features.{i}.block.{sub}", 1, 0, 1, act)
                sub += 1
            x = self._cbn(x, f"features.{i}.block.{sub}", stride, (k - 1) // 2, exp_c, act)
            sub += 1
            if use_se:
                x = self._se(x, f"features.{i}.block.{sub}")
                sub += 1
            x = self._cbn(x, f"features.{i}.block.{sub}", 1, 0, 1, None)
            if stride == 1 and in_c == out_c:
                x = x + identity

        x = self._cbn(x, "features.12", stride=1, padding=0, groups=1, act=hardswish)

        x = x.mean(axis=(1, 2))                                       # global avg pool
        x = self.sd["classifier.0.weight"] @ x + self.sd["classifier.0.bias"]
        x = hardswish(x)
        x = self.sd["classifier.3.weight"] @ x + self.sd["classifier.3.bias"]
        return x


# --------------------------------------------------------------------------
# 4. Preprocessing identical to the torchvision transform used at train time
# --------------------------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(frame_bgr):
    """cv2 BGR frame -> normalised (3,224,224) float32, matching
    transforms.Resize((224,224)) + ToTensor() + Normalize()."""
    from PIL import Image
    rgb = frame_bgr[:, :, ::-1]
    img = Image.fromarray(np.ascontiguousarray(rgb))
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)
