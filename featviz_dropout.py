"""Feature visualization on Lucid's InceptionV1 weights with dropout on intermediate layers during optimization.

Optimization is Lucid's exact recipe via Lucent (its PyTorch port): render_vis with its default
FFT + decorrelated image param, standard_transforms and Adam(0.05). Dropout is added on top via hooks.
Each row of the output grid uses a different dropout rate; each column is an independent image.
Usage: uv run featviz_dropout.py [--layer mixed4a] [--channel 476] [--rates 0 0.1 0.3 0.5] [--neuron]
"""
import argparse

import torch
import torchvision
from lucent.optvis import objectives, param, render

from top_channels import LAYERS, dev, load_model, model_input, zero_except

BLOCKS = [l for l in LAYERS if l.startswith("mixed")]  # mixed3a..mixed5b

p = argparse.ArgumentParser()
p.add_argument("--layer", default="mixed4a", choices=BLOCKS)
p.add_argument("--channel", type=int, default=476)
p.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5])
p.add_argument("--n", type=int, default=4, help="images per rate")
p.add_argument("--steps", type=int, default=512)
p.add_argument("--size", type=int, default=128, help="Lucid's default is 128 (upsampled to 224 before the model)")
p.add_argument("--channelwise", action="store_true", help="drop whole channels (dropout2d) instead of single neurons")
p.add_argument("--neuron", action="store_true", help="optimize only the center position of the channel instead of its mean")
p.add_argument("--keep", action="append", default=[], metavar="LAYER:CH,CH",
               help="zero every channel of LAYER except these (repeatable; 'LAYER:' zeroes all)")
p.add_argument("--out", default=None)
args = p.parse_args()
if not all(0 <= r < 1 for r in args.rates):
    p.error("--rates must be in [0, 1)")  # rate 1 would divide by zero below

model = load_model()
B = len(args.rates) * args.n
rates = torch.tensor(args.rates, device=dev).repeat_interleave(args.n).view(B, 1, 1, 1)

# Dropout hooks on every inception block before the target; a fresh mask is drawn every forward pass.
# Registered before Lucent's own hooks, so those see the dropped-out activations.
def drop(_, __, out):
    shape = out.shape[:2] + (1, 1) if args.channelwise else out.shape
    keep = (torch.rand(shape, device=dev) >= rates).float()
    return out * keep / (1 - rates)
for name in BLOCKS[: BLOCKS.index(args.layer)]:
    getattr(model, name).register_forward_hook(drop)
ablations = []
for spec in args.keep:
    layer, chs = spec.split(":")
    if layer not in LAYERS[: LAYERS.index(args.layer)]:
        p.error(f"--keep layer {layer} must come before --layer {args.layer}")
    ablations.append(getattr(model, layer).register_forward_hook(zero_except([int(c) for c in chs.split(",") if c])))

obj = (objectives.neuron if args.neuron else objectives.channel)(args.layer, args.channel)
imgs = render.render_vis(
    model, obj, lambda: param.image(args.size, batch=B),
    thresholds=(args.steps,), show_image=False,
)[-1]
img = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(dev)

# Final activation of every image measured without dropout, so rows are comparable.
# With --keep, measured both with the ablation (what was optimized) and on the intact network.
acts = {}
getattr(model, args.layer).register_forward_hook(lambda m, i, o: acts.__setitem__("t", o))
rates.zero_()
for label in (["ablated", "intact"] if ablations else ["clean"]):
    with torch.no_grad():
        model(model_input(img))
    a = acts["t"][:, args.channel]
    final = a[:, a.shape[1] // 2, a.shape[2] // 2] if args.neuron else a.mean(dim=(1, 2))
    for r, v in zip(args.rates, final.view(len(args.rates), args.n)):
        print(f"dropout {r:.2f}: {label} activation per image {[round(t, 2) for t in v.tolist()]}")
    for h in ablations:
        h.remove()

out = args.out or f"featviz_{args.layer}_ch{args.channel}{'_neuron' if args.neuron else ''}{'_ch2d' if args.channelwise else ''}{'_ablated' if args.keep else ''}.png"
torchvision.utils.save_image(img, out, nrow=args.n, padding=4, pad_value=1.0)
print("saved", out)
