import argparse
import os
import random
import shutil
from copy import deepcopy
from io import BytesIO

import numpy as np
import sklearn.metrics as sk
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import Dataset

from .augmentations import DataAugmentationDINO


SEED = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError("cumsum was found to be unstable: its last element does not correspond to sum")
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    if pos_label is None:
        pos_label = 1.0

    y_true = y_true == pos_label

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = (
        np.r_[recall[sl], 1],
        np.r_[fps[sl], 0],
        np.r_[tps[sl], 0],
        thresholds[sl],
    )
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / np.sum(~y_true), thresholds[cutoff]


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[: len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr, _threshold = fpr_and_fdr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr


def find_best_threshold(y_true, y_pred):
    n = y_true.shape[0]
    if y_pred[0 : n // 2].max() <= y_pred[n // 2 : n].min():
        return (y_pred[0 : n // 2].max() + y_pred[n // 2 : n].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / n
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc
    return best_thres


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format="jpeg", quality=quality)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
    return Image.fromarray(img)


def add_phase_noise(img, mean=0.0, std=0.1):
    img = np.array(img).astype(np.float32)
    f_transform = np.fft.fft2(img)
    magnitude = np.abs(f_transform)
    phase = np.angle(f_transform)
    noise = np.random.normal(mean, std, phase.shape)
    phase_noisy = phase + noise
    f_transform_noisy = magnitude * np.exp(1j * phase_noisy)
    img_noisy = np.fft.ifft2(f_transform_noisy)
    img_noisy = np.abs(img_noisy)
    img_noisy = np.clip(img_noisy, 0, 255)
    return Image.fromarray(img_noisy.astype(np.uint8))


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def calculate_cosine_similarity(tensor1, tensor2):
    return torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=-1)


def recursively_read(rootdir, must_contain, exts=("png", "jpg", "jpeg", "bmp")):
    out = []
    for r, _d, files in os.walk(rootdir):
        for file in files:
            parts = file.rsplit(".", 1)
            if len(parts) != 2:
                continue
            if parts[1].lower() in exts and must_contain in os.path.join(r, file):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=""):
    return recursively_read(path, must_contain)


class RealFakeDataset(Dataset):
    def __init__(
        self,
        real_path,
        fake_path,
        data_mode,
        max_sample,
        jpeg_quality=None,
        gaussian_sigma=None,
        phase_noise_std=None,
        trans=None,
    ):
        assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.phase_noise_std = phase_noise_std
        self.transform = trans

        real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        self.total_list = real_list + fake_list

        self.labels_dict = {p: 0 for p in real_list}
        self.labels_dict.update({p: 1 for p in fake_list})

    def read_path(self, real_path, fake_path, data_mode, max_sample):
        if data_mode == "wang2020":
            real_list = get_list(real_path, must_contain="0_real")
            fake_list = get_list(fake_path, must_contain="1_fake")
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        if max_sample == 0:
            real_list = random.sample(real_list, 1000)
            fake_list = random.sample(fake_list, 1000)
        elif max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = min(100, len(real_list), len(fake_list))
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[:max_sample]
            fake_list = fake_list[:max_sample]
        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)
        if self.phase_noise_std is not None:
            img = add_phase_noise(img, mean=0.0, std=self.phase_noise_std)

        img = self.transform(img)
        return img, label


def add_noise_to_transformer_layers(model, std_dev_ratio, num_layers=15):
    layer_count = 0
    for name, param in model.named_parameters():
        if "blocks" in name:
            layer_count += 1
            if layer_count <= num_layers * 14 and param.requires_grad:
                noise = torch.randn_like(param) * std_dev_ratio * torch.mean(param)
                param.data.add_(noise)


@torch.no_grad()
def validate(original_model, noise_model, loader, crops, device):
    original_model.eval()
    noise_model.eval()

    score_in = []
    score_out = []
    y_true = []
    y_pred_similarity = []
    y_pred_fake_score = []
    y_labels = []

    for img, labels in loader:
        labels = labels.to(device)
        y_true.append(labels.detach().cpu().numpy())

        x = img["source1"].to(device)
        original_embedding_before_norm = original_model(x)
        original_embedding = original_embedding_before_norm / original_embedding_before_norm.norm(dim=-1, keepdim=True)

        l2_distances = []
        for _ in range(crops):
            noisy_embedding_before_norm = noise_model(x)
            noisy_embedding = noisy_embedding_before_norm / noisy_embedding_before_norm.norm(dim=-1, keepdim=True)
            similarity = calculate_cosine_similarity(original_embedding, noisy_embedding)
            l2_distances.append(similarity.detach().cpu().numpy())

        distance = np.stack(l2_distances, axis=1)
        cosine_similarity = np.mean(distance, axis=1)

        y_pred_similarity.append(cosine_similarity)
        y_pred_fake_score.extend(1 - cosine_similarity)
        y_labels.extend(labels.detach().cpu().numpy().flatten().tolist())

        for i in range(labels.shape[0]):
            if labels[i].item() == 1:
                score_in.append(cosine_similarity[i])
            else:
                score_out.append(cosine_similarity[i])

    y_labels = np.array(y_labels)
    y_pred_fake_score = np.array(y_pred_fake_score)

    best_thres = find_best_threshold(y_labels, y_pred_fake_score)
    _r_acc, _f_acc, acc = calculate_acc(y_labels, y_pred_fake_score, best_thres)

    y_true = np.concatenate(y_true)
    y_pred_similarity = np.concatenate(y_pred_similarity)

    ap = average_precision_score(1 - y_true, y_pred_similarity)
    auroc, _aupr, _fpr = get_measures(score_out, score_in)

    return auroc, ap, acc


def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--real_path", type=str, default="datasets/test/san/0_real")
    parser.add_argument("--fake_path", type=str, default="datasets/test/san/1_fake")
    parser.add_argument("--data_mode", type=str, default="ours", choices=["wang2020", "ours"])
    parser.add_argument("--max_sample", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--result_folder", type=str, default="result_noise_model_detect")

    parser.add_argument("--jpeg_quality", type=int, default=None)
    parser.add_argument("--gaussian_sigma", type=float, default=None)
    parser.add_argument("--phase_noise_std", type=float, default=None)

    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_fp16", action="store_true", default=False)

    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--noise_layers", type=int, default=19)
    parser.add_argument("--crops", type=int, default=1)

    opt = parser.parse_args(argv)

    set_seed(opt.seed)

    device = opt.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder, exist_ok=True)

    original_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    noise_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")

    add_noise_to_transformer_layers(noise_model, opt.noise_std, opt.noise_layers)

    original_model = original_model.to(device)
    noise_model = noise_model.to(device)

    if opt.use_fp16 and device.startswith("cuda"):
        original_model = original_model.half()
        noise_model = noise_model.half()

    data_transform = DataAugmentationDINO(
        (0.9, 1.0),
        (0.05, 0.4),
        opt.crops,
        global_crops_size=224,
        local_crops_size=96,
    )

    dataset = RealFakeDataset(
        opt.real_path,
        opt.fake_path,
        opt.data_mode,
        opt.max_sample,
        jpeg_quality=opt.jpeg_quality,
        gaussian_sigma=opt.gaussian_sigma,
        phase_noise_std=opt.phase_noise_std,
        trans=data_transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    auroc, ap, acc = validate(original_model, noise_model, loader, opt.crops, device)

    out_path = os.path.join(opt.result_folder, "metrics.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"AUROC: {auroc}\n")
        f.write(f"AP: {ap}\n")
        f.write(f"ACC: {acc}\n")

    print(f"AUROC: {auroc}")
    print(f"AP: {ap}")
    print(f"ACC: {acc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

