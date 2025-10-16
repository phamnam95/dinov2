import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dinov2.models import vision_transformer as vits


def load_npy_as_tensor(path: Path, is_3d: bool, channels_last: bool) -> torch.Tensor:
    arr = np.load(str(path))
    # Ensure float32
    arr = arr.astype(np.float32, copy=False)
    # Replace NaNs with zeros to avoid propagation
    np.nan_to_num(arr, copy=False)
    t = torch.from_numpy(arr)
    # Add channel dimension if missing
    if is_3d:
        # expected [C,D,H,W] or [D,H,W] or [D,H,W,C]
        if t.dim() == 3:  # [D,H,W]
            t = t.unsqueeze(0)
        elif t.dim() == 4 and channels_last:  # [D,H,W,C] -> [C,D,H,W]
            t = t.permute(3, 0, 1, 2)
        elif t.dim() == 4:  # assume [C,D,H,W]
            pass
        else:
            raise ValueError(f"Unsupported 3D array shape for {path}: {tuple(t.shape)}")
    else:
        # expected [C,H,W] or [H,W] or [H,W,C]
        if t.dim() == 2:  # [H,W]
            t = t.unsqueeze(0)
        elif t.dim() == 3 and channels_last:  # [H,W,C] -> [C,H,W]
            t = t.permute(2, 0, 1)
        elif t.dim() == 3:  # [C,H,W]
            pass
        else:
            raise ValueError(f"Unsupported 2D array shape for {path}: {tuple(t.shape)}")
    return t


def adapt_2d_state_to_3d(model: torch.nn.Module, state: dict) -> dict:
    """Inflate 2D ViT weights to 3D for PatchEmbed and positional embeddings."""
    new_state = dict(state)
    # patch embed
    pe_w_key = "patch_embed.proj.weight"
    if pe_w_key in state and state[pe_w_key].ndim == 4:
        w2d = state[pe_w_key]  # [E, Cin, Kh, Kw]
        kd = model.patch_embed.proj.weight.shape[2]
        w3d = w2d.unsqueeze(2).repeat(1, 1, kd, 1, 1) / kd
        new_state[pe_w_key] = w3d
    # pos embed
    pos_key = "pos_embed"
    if pos_key in state:
        pos2d = state[pos_key]
        if pos2d.ndim == 3 and pos2d.shape[0] == 1:
            cls_pos = pos2d[:, :1, :]
            patch_pos = pos2d[:, 1:, :]
            n2 = patch_pos.shape[1]
            c = patch_pos.shape[2]
            m = int(np.sqrt(n2))
            if m * m == n2 and hasattr(model.patch_embed, "patches_resolution") and len(model.patch_embed.patches_resolution) == 3:
                d0, h0, w0 = model.patch_embed.patches_resolution
                patch_pos_2d = patch_pos.view(1, m, m, c).permute(0, 3, 1, 2)  # [1,C,M,M]
                patch_pos_3d = F.interpolate(
                    patch_pos_2d.unsqueeze(2), size=(d0, h0, w0), mode="trilinear", align_corners=False
                ).squeeze(2)
                patch_pos_flat = patch_pos_3d.permute(0, 2, 3, 1).reshape(1, d0 * h0 * w0, c)
                new_state[pos_key] = torch.cat([cls_pos, patch_pos_flat], dim=1)
    return new_state


def build_model(arch: str, patch_size: int, is_3d: bool, num_register_tokens: int = 0, img_size: int = 224) -> torch.nn.Module:
    fn = getattr(vits, arch)
    model = fn(patch_size=patch_size, num_register_tokens=num_register_tokens, is_3d=is_3d, img_size=img_size)
    model.eval()
    return model


def load_weights(model: torch.nn.Module, weights_path: str, is_3d: bool) -> None:
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("teacher") or ckpt.get("model") or ckpt
    # Adapt if needed
    if is_3d:
        # Detect 2D kernel
        pe_w_key = "patch_embed.proj.weight"
        if pe_w_key in state and state[pe_w_key].ndim == 4:
            state = adapt_2d_state_to_3d(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)}")


def collect_files_by_class(root: str, max_files_per_class: int = 0) -> Tuple[List[str], List[int], List[str]]:
    root_p = Path(root)
    classes = sorted([d.name for d in root_p.iterdir() if d.is_dir()])
    file_paths: List[str] = []
    labels: List[int] = []
    class_names: List[str] = []
    for ci, cname in enumerate(classes):
        npys = sorted((root_p / cname).glob("*.npy"))
        if max_files_per_class > 0:
            npys = npys[: max_files_per_class]
        for p in npys:
            file_paths.append(str(p))
            labels.append(ci)
        class_names.append(cname)
    return file_paths, labels, class_names


def compute_embeddings(
    model: torch.nn.Module,
    files: List[str],
    labels: List[int],
    device: str,
    batch_size: int,
    is_3d: bool,
    channels_last_npy: bool,
    use_cls: bool = True,
    use_patch_embed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[torch.Tensor] = []
    ys: List[int] = []
    with torch.no_grad():
        for i, fp in enumerate(files):
            t = load_npy_as_tensor(Path(fp), is_3d=is_3d, channels_last=channels_last_npy)
            xs.append(t)
            ys.append(labels[i])
            if len(xs) == batch_size or i == len(files) - 1:
                xbatch = torch.stack(xs, dim=0).to(device, non_blocking=True)
                if use_patch_embed:
                    # Extract patch embedding before transformer blocks
                    with torch.no_grad():
                        pe = model.patch_embed(xbatch)
                        # Global average over tokens
                        feats = pe.mean(dim=1)
                else:
                    out = model(xbatch, is_training=True)
                    feats = out["x_norm_clstoken"] if use_cls else out["x_norm_patchtokens"].mean(dim=1)
                feats = F.normalize(feats, dim=-1)
                feats_np = feats.cpu().numpy()
                if i == len(files) - 1 or len(xs) == batch_size:
                    if i == batch_size - 1:
                        emb = feats_np
                        ys_np = np.array(ys, dtype=np.int64)
                    else:
                        try:
                            emb = np.vstack((emb, feats_np))  # type: ignore[name-defined]
                            ys_np = np.concatenate((ys_np, np.array(ys, dtype=np.int64)))  # type: ignore[name-defined]
                        except NameError:
                            emb = feats_np
                            ys_np = np.array(ys, dtype=np.int64)
                xs.clear()
                ys.clear()
    return emb, ys_np


def project_to_3d(emb: np.ndarray, method: str = "umap", random_state: int = 0) -> np.ndarray:
    if method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=3, random_state=random_state)
            return reducer.fit_transform(emb)
        except Exception:
            pass
    if method.lower() == "tsne":
        proj = TSNE(n_components=3, random_state=random_state, init="pca", learning_rate="auto").fit_transform(emb)
        return proj
    # fallback to PCA
    proj = PCA(n_components=3, random_state=random_state).fit_transform(emb)
    return proj


def compute_metrics(emb: np.ndarray, labels: np.ndarray, sample_limit: int = 10000, random_state: int = 0) -> dict:
    n = emb.shape[0]
    if n > sample_limit:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_limit, replace=False)
        E = emb[idx]
        y = labels[idx]
    else:
        E = emb
        y = labels
    metrics = {}
    # Silhouette can be expensive; try/catch
    try:
        metrics["silhouette"] = float(silhouette_score(E, y, metric="euclidean"))
    except Exception as e:
        metrics["silhouette"] = None
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(E, y))
    except Exception:
        metrics["davies_bouldin"] = None
    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(E, y))
    except Exception:
        metrics["calinski_harabasz"] = None
    return metrics


def save_3d_plot(proj3d: np.ndarray, labels: np.ndarray, class_names: List[str], out_path: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    num_classes = len(class_names)
    cmap = plt.get_cmap("tab20")
    for ci in range(num_classes):
        idx = labels == ci
        ax.scatter(proj3d[idx, 0], proj3d[idx, 1], proj3d[idx, 2], s=6, alpha=0.7, color=cmap(ci % 20), label=class_names[ci])
    ax.set_title("Embedding Clusters (3D)")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Embedding clustering evaluation")
    parser.add_argument("--data-root", required=True, help="Root folder with class subfolders containing .npy files")
    parser.add_argument("--weights", required=True, help="Path to pretrained weights (teacher/model or full state dict)")
    parser.add_argument("--arch", default="vit_large", help="ViT arch: vit_small|vit_base|vit_large|vit_giant2")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--is-3d", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channels-last-npy", action="store_true", help="Set if npy arrays are channels-last")
    parser.add_argument("--use-cls", action="store_true", help="Use CLS token embedding (default). If false, avg patch tokens.")
    parser.add_argument("--use-patch-embed", action="store_true", help="Use patch embedding (pre-transformer) averaged over tokens")
    parser.add_argument("--projection", default="umap", choices=["umap", "tsne", "pca"])
    parser.add_argument("--metrics-sample-limit", type=int, default=10000)
    parser.add_argument("--max-files-per-class", type=int, default=0)
    parser.add_argument("--output-dir", default="eval_embed_out")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build and load model
    model = build_model(args.arch, args.patch_size, args.is_3d, img_size=args.img_size)
    load_weights(model, args.weights, is_3d=args.is_3d)
    model.to(device)
    model.eval()

    # Collect files
    files, labels, class_names = collect_files_by_class(args.data_root, args.max_files_per_class)
    print(f"Found {len(files)} files across {len(class_names)} classes")

    # Compute embeddings
    emb, y_np = compute_embeddings(
        model,
        files,
        labels,
        device=str(device),
        batch_size=args.batch_size,
        is_3d=args.is_3d,
        channels_last_npy=args.channels_last_npy,
        use_cls=True if args.use_cls else True,
        use_patch_embed=bool(args.use_patch_embed),
    )

    # Project to 3D and plot
    proj3d = project_to_3d(emb, method=args.projection, random_state=args.seed)
    plot_path = os.path.join(args.output_dir, "embed_3d.png")
    save_3d_plot(proj3d, y_np, class_names, plot_path)

    # Metrics
    metrics = compute_metrics(emb, y_np, sample_limit=args.metrics_sample_limit, random_state=args.seed)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved:", plot_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
