"""Local web app for top-down circuit pruning on Lucid's InceptionV1.

Start at a root channel, expand any node to see the top-k channels of the previous layer by attribution to it,
tick the ones to keep, and keep descending as far as you want. In every layer where you made a selection, every
channel that was not selected (in any branch) is switched off. "Gerar" renders the root channel with those
ablations (via featviz_dropout.py --keep).

Usage: uv run circuit_app.py [--layer mixed4e] [--channel 55] [--k 30] [--port 8000]  ->  open http://127.0.0.1:8000
"""
import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torchvision

from top_channels import LAYERS, load_model, rank, visualize

p = argparse.ArgumentParser()
p.add_argument("--layer", default="mixed4e")
p.add_argument("--channel", type=int, default=55)
p.add_argument("--k", type=int, default=30)
p.add_argument("--port", type=int, default=8000)
args = p.parse_args()
assert args.layer in LAYERS[3:], f"--layer must be one of {LAYERS[3:]}"  # featviz_dropout.py renders mixed* targets only
HERE = Path(__file__).parent
LOCK = threading.Lock()  # one GPU job at a time; the model and its hooks are shared


def png(t):
    buf = io.BytesIO()
    torchvision.transforms.functional.to_pil_image(t).save(buf, "PNG")
    return buf.getvalue()


model = load_model()
thumbs = {}      # (layer, ch) -> png of that channel's feature visualization
expansions = {}  # (layer, ch) -> top-k channels of the previous layer by attribution
references = {}  # (layer, ch) -> png grid of the channel's visualizations on the intact network


# shortcut: caches live in memory only, so a restart recomputes everything; persist to disk if sessions get long.
def expand(layer, ch):
    key = (layer, ch)
    if key not in expansions:
        prev, target, attr = rank(model, layer, ch)
        thumbs.setdefault(key, png(target[0]))
        references[key] = png(torchvision.utils.make_grid(target, nrow=len(target), padding=4, pad_value=1.0))
        top = attr.argsort(descending=True)[: args.k].tolist()
        new = [c for c in top if (prev, c) not in thumbs]
        if new:
            thumbs.update({(prev, c): png(t) for c, t in zip(new, visualize(model, prev, new))})
        expansions[key] = {"prev": prev, "total": len(attr),
                           "channels": [{"ch": c, "attr": round(attr[c].item(), 2)} for c in top]}
        torch.cuda.empty_cache()
    return expansions[key]


def generate(keep):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "final.png"
        cmd = [sys.executable, str(HERE / "featviz_dropout.py"), "--layer", args.layer, "--channel", str(args.channel),
               "--rates", "0", "--n", "4", "--out", str(out)]
        for layer, chs in keep.items():
            cmd += ["--keep", f"{layer}:{','.join(map(str, chs))}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        return {"image": base64.b64encode(out.read_bytes()).decode(),
                "log": [l for l in r.stdout.splitlines() if "activation" in l]}


print("ranking the root's inputs and rendering visualizations...", flush=True)
expand(args.layer, args.channel)


class Handler(BaseHTTPRequestHandler):
    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        parts = self.path.split("/")[1:]
        try:
            if self.path == "/":
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/meta":
                self.send_json({"layer": args.layer, "channel": args.channel, "layers": LAYERS, "k": args.k})
            elif self.path == "/reference.png":
                self.send(references[(args.layer, args.channel)], "image/png")
            elif len(parts) == 3 and parts[0] == "img" and (parts[1], int(parts[2].removesuffix(".png"))) in thumbs:
                self.send(thumbs[(parts[1], int(parts[2].removesuffix(".png")))], "image/png")
            elif len(parts) == 3 and parts[0] == "expand" and parts[1] in LAYERS[1:]:
                with LOCK:
                    self.send_json(expand(parts[1], int(parts[2])))
            else:
                self.send(b"not found", "text/plain", 404)
        except Exception as e:
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        if self.path != "/generate":
            return self.send(b"not found", "text/plain", 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            keep = {l: sorted({int(c) for c in chs}) for l, chs in body["keep"].items()}
            assert all(l in LAYERS for l in keep), "unknown layer"
            with LOCK:
                self.send_json(generate(keep))
        except Exception as e:
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)


PAGE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Poda de circuito</title>
<style>
:root { --bg:#f6f5f2; --card:#fff; --fg:#1d1d1b; --muted:#6b6a66; --line:#dcdad4; --accent:#c2410c; --on:#fff; --warn:#a16207; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --card:#222220; --fg:#ecebe7; --muted:#a3a29c; --line:#3a3936; --accent:#fb923c; --on:#161615; --warn:#facc15; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:0 0 8px; } .muted { color:var(--muted); }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; margin-top:16px; overflow-x:auto; }
button { font:inherit; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:8px; padding:6px 12px; cursor:pointer; }
button:focus-visible, input:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
button.primary { background:var(--accent); color:var(--on); border-color:var(--accent); } button:disabled { opacity:.5; cursor:wait; }
button.thumb { padding:0; border:none; background:none; border-radius:6px; line-height:0; flex:none; }
button.thumb img { border-radius:6px; display:block; }
ul.tree, ul.tree ul { list-style:none; margin:0; padding-left:28px; border-left:2px solid var(--line); }
ul.tree { padding-left:0; border:none; }
ul.tree li { margin:10px 0; }
.node { display:flex; gap:12px; align-items:center; }
.info { display:flex; flex-direction:column; gap:4px; align-items:flex-start; }
.badge { font-size:12px; color:var(--warn); }
.sel { margin:10px 0 4px; padding:12px; border:1px dashed var(--line); border-radius:10px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(124px, 1fr)); gap:10px; margin:10px 0; }
.cell { border:2px solid transparent; border-radius:8px; padding:4px; } .cell.on { border-color:var(--accent); }
.cell img { width:100%; height:auto; } .cell label { display:flex; gap:6px; align-items:center; font-size:13px; margin-top:4px; cursor:pointer; }
.cell input { accent-color:var(--accent); width:16px; height:16px; }
.row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.layer { margin:6px 0; } .chips { font-size:13px; color:var(--muted); }
dialog { border:none; border-radius:12px; padding:12px; background:var(--card); color:var(--fg); max-width:min(560px, 94vw); }
dialog::backdrop { background:rgba(0,0,0,.6); } dialog img { width:100%; image-rendering:auto; border-radius:8px; display:block; }
.result img { width:100%; border-radius:8px; margin-top:6px; } pre { font-size:12px; white-space:pre-wrap; color:var(--muted); }
</style></head><body><main>
<h1 id="title">Poda de circuito</h1>
<p class="muted">Expanda um nó para ver o top <span id="k"></span> da camada anterior por atribuição a ele e marque quais manter.
Em toda camada onde você fez uma seleção, os canais não selecionados (em nenhum ramo) são desligados. Clique numa imagem para ampliar.</p>
<div class="card"><h2>Árvore</h2><ul class="tree" id="tree"></ul></div>
<div class="card"><h2>O que vai ser desligado</h2><div id="summary"></div>
  <div class="row" style="margin-top:12px"><button class="primary" id="go">Gerar imagem da raiz</button></div></div>
<div class="card result" id="result" hidden>
  <strong>Raiz sem ablação (referência)</strong><img src="/reference.png" alt="Visualização da raiz na rede intacta">
  <strong style="display:block;margin-top:12px">Raiz com os canais não selecionados desligados</strong><img id="final" alt="Visualização da raiz com a poda">
  <pre id="log"></pre></div>
<dialog id="zoom"><img id="zimg" alt=""><p id="zcap" style="margin:8px 0 0"></p></dialog>
</main><script>
let meta, root; const totals = {}; const $ = id => document.getElementById(id);
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const prevOf = l => meta.layers[meta.layers.indexOf(l) - 1];
const mk = (layer, ch, attr) => ({ layer, ch, attr, kids: null, cand: null, sel: null, loading: false });

function keepMap() {  // layer -> Set of kept channels, for every layer where some node was expanded
  const m = {};
  (function walk(n) {
    if (!n.kids) return;
    const p = prevOf(n.layer); m[p] ??= new Set();
    n.kids.forEach((k, ch) => { m[p].add(ch); walk(k); });
  })(root);
  return m;
}
function zoom(layer, ch, attr) {
  $("zimg").src = `/img/${layer}/${ch}.png`; $("zimg").alt = `Visualização de ${layer}:${ch}`;
  $("zcap").textContent = `${layer}:${ch}` + (attr != null ? ` · atribuição ${attr}` : ""); $("zoom").showModal();
}
$("zoom").onclick = () => $("zoom").close();
const thumb = (layer, ch, size, attr) => el("button", { className: "thumb", title: "Ampliar", onclick: () => zoom(layer, ch, attr) },
  el("img", { src: `/img/${layer}/${ch}.png`, width: size, height: size, alt: `Visualização de ${layer}:${ch}`, loading: "lazy" }));

async function expand(n) {
  if (!n.cand) {
    n.loading = true; render();
    try {
      const j = await (await fetch(`/expand/${n.layer}/${n.ch}`)).json();
      if (j.error) throw new Error(j.error);
      n.cand = j; totals[j.prev] = j.total;
    } catch (e) { alert("Erro: " + e.message); }
    n.loading = false;
  }
  if (n.cand) n.sel = new Set(n.kids ? n.kids.keys() : []);
  render();
}
function selPanel(n) {
  const grid = el("div", { className: "grid" });
  for (const c of n.cand.channels) {
    const cb = el("input", { type: "checkbox", checked: n.sel.has(c.ch) });
    const cell = el("div", { className: "cell" + (cb.checked ? " on" : "") }, thumb(n.cand.prev, c.ch, 120, c.attr),
      el("label", {}, cb, `${c.ch} · ${c.attr}`));
    cb.onchange = () => { cb.checked ? n.sel.add(c.ch) : n.sel.delete(c.ch); cell.classList.toggle("on", cb.checked); };
    grid.append(cell);
  }
  const ok = el("button", { className: "primary", textContent: "Confirmar seleção", onclick: () => {
    n.kids = new Map(n.cand.channels.filter(c => n.sel.has(c.ch)).map(c => [c.ch, n.kids?.get(c.ch) ?? mk(n.cand.prev, c.ch, c.attr)]));
    n.sel = null; render();
  } });
  const cancel = el("button", { textContent: "Cancelar", onclick: () => { n.sel = null; render(); } });
  return el("div", { className: "sel" },
    el("div", { textContent: `Entradas de ${n.layer}:${n.ch}: top ${n.cand.channels.length} de ${n.cand.total} canais da ${n.cand.prev} (número = canal · atribuição)` }),
    grid, el("div", { className: "row" }, ok, cancel));
}
function renderNode(n, keep) {
  const p = prevOf(n.layer);
  const info = el("div", { className: "info" }, el("strong", { textContent: `${n.layer}:${n.ch}` }));
  if (n.attr != null) info.append(el("span", { className: "muted", textContent: `atribuição ao pai: ${n.attr}` }));
  if (!p) info.append(el("span", { className: "muted", textContent: "primeira camada" }));
  else if (n.loading) info.append(el("span", { className: "muted", textContent: `calculando top ${meta.k} da ${p}... (~1 min)` }));
  else if (!n.sel) {
    if (n.kids) info.append(el("span", { className: "muted", textContent: `${n.kids.size} entradas mantidas na ${p}` }));
    else if (keep[p]) info.append(el("span", { className: "badge", textContent: `não expandido: só recebe os ${keep[p].size} canais mantidos na ${p}` }));
    info.append(el("button", { textContent: n.kids ? "Editar seleção" : `Expandir (top ${meta.k} da ${p})`, onclick: () => expand(n) }));
  }
  const li = el("li", {}, el("div", { className: "node" }, thumb(n.layer, n.ch, 72, n.attr), info));
  if (n.sel) li.append(selPanel(n));
  if (n.kids?.size) li.append(el("ul", {}, ...[...n.kids.values()].map(k => renderNode(k, keep))));
  return li;
}
function render() {
  const keep = keepMap();
  $("tree").replaceChildren(renderNode(root, keep));
  const layers = meta.layers.filter(l => keep[l]).reverse();
  $("summary").replaceChildren(...(layers.length ? layers.map(l => el("div", { className: "layer" },
    el("strong", { textContent: `${l}: ` }),
    `${keep[l].size} mantidos, ${totals[l] - keep[l].size} desligados`,
    keep[l].size ? "" : el("span", { className: "badge", textContent: "  (todos desligados: a raiz deve ir a zero)" }),
    el("div", { className: "chips", textContent: keep[l].size ? "mantidos: " + [...keep[l]].sort((a, b) => a - b).join(", ") : "" })))
    : [el("p", { className: "muted", textContent: "Nada ainda: expanda a raiz e confirme uma seleção." })]));
}
$("go").onclick = async () => {
  $("go").disabled = true; $("go").textContent = "Gerando... (~15 s)";
  try {
    const keep = Object.fromEntries(Object.entries(keepMap()).map(([l, s]) => [l, [...s]]));
    const j = await (await fetch("/generate", { method: "POST", body: JSON.stringify({ keep }) })).json();
    if (j.error) throw new Error(j.error);
    $("final").src = "data:image/png;base64," + j.image; $("log").textContent = j.log.join("\\n");
    $("result").hidden = false; $("result").scrollIntoView({ behavior: "smooth" });
  } catch (e) { alert("Erro: " + e.message); }
  $("go").disabled = false; $("go").textContent = "Gerar imagem da raiz";
};
fetch("/meta").then(r => r.json()).then(m => {
  meta = m; root = mk(m.layer, m.channel, null);
  $("title").textContent = `Poda de circuito a partir de ${m.layer}:${m.channel}`; $("k").textContent = m.k;
  render();
});
</script></body></html>"""

print(f"open http://127.0.0.1:{args.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
