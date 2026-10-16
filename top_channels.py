"""Shared helpers for Lucid's InceptionV1 (via Lucent), plus a CLI that ranks the channels of the layer right
before a target channel by attribution (activation x gradient) on the target's own feature visualization,
and saves the top-k of them as a labeled grid.

No dropout anywhere. Usage: uv run top_channels.py [--layer mixed4e] [--channel 55] [--k 30]
"""
import argparse
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from lucent.modelzoo import inceptionv1
from lucent.optvis import objectives, param, render
from PIL import ImageDraw

dev = "cuda" if torch.cuda.is_available() else "cpu"
# InceptionV1 layers in forward order, as named by Lucent.
LAYERS = ["conv2d0", "conv2d1", "conv2d2"] + [f"mixed{b}" for b in ("3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b")]


def load_model():
    torch.manual_seed(0)
    np.random.seed(0)  # Lucent's transforms sample with numpy
    # Lucid's InceptionV1 (inception5h), ported to PyTorch; same channel indices as Lucid / OpenAI Microscope.
    model = inceptionv1(pretrained=True).eval().to(dev)
    for q in model.parameters():
        q.requires_grad_(False)
    return model


def model_input(img):
    """[0,1] images -> what InceptionV1 takes, exactly as render_vis prepares them (upsample to >=224, [-117, 138])."""
    return F.interpolate(img, size=max(224, img.shape[-1]), mode="bilinear", align_corners=True) * 255 - 117


def zero_except(keep):
    """Forward hook that zeroes every channel of the module's output except those in `keep`."""
    def hook(_, __, out):
        mask = torch.zeros(out.shape[1], device=out.device)
        mask[keep] = 1
        return out * mask.view(1, -1, 1, 1)
    return hook


def clear_hooks(model):
    # Lucent's render_vis registers ~138 forward hooks per call and never removes them; in a long-lived
    # process they pile up and slow every forward pass. Also removes any other hook on the model.
    for m in model.modules():
        m._forward_hooks.clear()


def rank(model, layer, channel, n=4):
    """Returns (previous layer name, target visualizations (n,3,128,128) on cpu, attribution per previous channel)."""
    assert LAYERS.index(layer) > 0, f"{layer} has no previous layer"
    prev = LAYERS[LAYERS.index(layer) - 1]
    # 1. Feature visualization of the target, without dropout.
    imgs = render.render_vis(model, objectives.channel(layer, channel),
                             lambda: param.image(128, batch=n), show_image=False, progress=False)[-1]
    clear_hooks(model)
    img = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(dev).requires_grad_()  # weights are frozen; builds the graph
    # 2. Attribution of every channel of the previous layer to the target's mean activation.
    acts = {}
    hooks = [getattr(model, b).register_forward_hook(lambda m, i, o, b=b: acts.__setitem__(b, o)) for b in (prev, layer)]
    model(model_input(img))
    a = acts[prev]
    g, = torch.autograd.grad(acts[layer][:, channel].mean(), a)
    for h in hooks:
        h.remove()
    return prev, img.detach().cpu(), (a * g).sum(dim=(2, 3)).mean(0).cpu()


def visualize(model, layer, channels, chunk=10):
    """Feature visualization of each channel, one image per channel, in batches of `chunk` (bounds GPU memory)."""
    out = []
    for s in range(0, len(channels), chunk):
        part = channels[s : s + chunk]
        obj = reduce(lambda x, y: x + y, [objectives.channel(layer, c, batch=i) for i, c in enumerate(part)])
        vis = render.render_vis(model, obj, lambda: param.image(128, batch=len(part)), show_image=False, progress=False)[-1]
        clear_hooks(model)
        out.append(torch.from_numpy(vis).permute(0, 3, 1, 2))
    return torch.cat(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", default="mixed4e")
    p.add_argument("--channel", type=int, default=55)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--n", type=int, default=4, help="target visualizations used for attribution")
    args = p.parse_args()

    model = load_model()
    prev, _, attr = rank(model, args.layer, args.channel, args.n)
    top = attr.argsort(descending=True)[: args.k].tolist()
    print(f"{prev}: {len(attr)} channels scored; top {args.k} by attribution to {args.layer}:{args.channel}")
    for r, c in enumerate(top):
        print(f"{r:2d}  {prev}:{c:<4d} attr {attr[c].item():9.1f}")

    # 3. Feature visualization of each top channel, as a labeled grid.
    grid = torchvision.utils.make_grid(visualize(model, prev, top), nrow=6, padding=18, pad_value=1.0)
    pil = torchvision.transforms.functional.to_pil_image(grid)
    draw = ImageDraw.Draw(pil)
    for i, c in enumerate(top):
        draw.text((18 + (i % 6) * 146, 3 + (i // 6) * 146), f"#{i} {prev}:{c}", fill=(0, 0, 0))
    out = f"top{args.k}_{prev}_for_{args.layer}_ch{args.channel}.png"
    pil.save(out)
    print("saved", out)
