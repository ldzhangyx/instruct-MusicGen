import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(sys.path[0], "../../../.."))


def tile(x, n):
    return torch.stack([x] * n, 0)


def generate_mask(tag, max_len):
    mask = torch.zeros([2, max_len])
    if tag == "non":
        return mask

    tag_dict = {
        "piano": 0,
        "drums": 1
    }
    for k in tag.split("-"):
        mask[tag_dict[k]] = 1
    return mask


def pad(x, max_len, dim, n_dim):
    if x.shape[dim] < max_len:
        if n_dim == 1:
            x = np.pad(x, (0, max_len - x.shape[dim]),
                       "constant", constant_values=(0, 0))
        elif n_dim == 2:
            if dim == 0:
                x = np.pad(x, ((0, max_len - x.shape[dim]), (0, 0)),
                           "constant", constant_values=(0, 0))
            elif dim == 1:
                x = np.pad(x, ((0, 0), (0, max_len - x.shape[dim])),
                           "constant", constant_values=(0, 0))
    return x


def detect_silence(path, mode="wav"):
    x = AudioSegment.from_file(path, mode)
    dBFS = x.dBFS
    sil = silence.detect_silence(x, min_silence_len=1000, silence_thresh=dBFS - 16)
    if len(sil) == 0:
        return 0, -1
    x_len = x.duration_seconds
    st = 0 if sil[0][0] > 0 else sil[0][1] / 1000.
    ed = sil[-1][0] / 1000. if sil[-1][1] / 1000. >= x_len - 1 else -1
    return st, ed


def np2torch(x):
    return torch.from_numpy(x)


def save_audio(path, x, sample_rate):
    audio_write(path, x.cpu(), sample_rate, strategy="loudness", loudness_compressor=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def listdir(folder, suffix=None):
    outs = []
    for f in os.listdir(folder):
        if suffix is None or str.endswith(f, suffix):
            outs.append(f)
    return outs


def read_lst(path, split=None):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [al.rstrip() for al in lines]
    if split is None:
        return lines
    lines = [al.split(split) for al in lines]
    return lines


def save_lst(path, data):
    with open(path, "w") as f:
        f.writelines("\n".join(data))


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for k, param in model.named_parameters():

        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            print(k)
            trainable_params += num_params

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
def freeze(model):
    for n, p in model.named_parameters():
        p.requires_grad = False


def unfreeze(model):
    for n, p in model.named_parameters():
        p.requires_grad = True