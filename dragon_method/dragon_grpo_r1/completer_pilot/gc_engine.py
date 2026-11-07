"""vLLM calls for the pilot: DemandEngine plus one max_tokens per prompt, on any prompt in the item."""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demand_rl"))
import vllm_backend as VB  # noqa: E402

VC = VB.VC


def item_images(it: Dict):
    """The item's images in prompt order: the diagram, plus the crop sheet for a v3 completer item.
    it["ablate"] (eval only): "blank_sheet" / "blank_diagram" replace that image by white of the same size,
    so the prompt's token count does not change; "blank_visual" blanks both. it["marks"] (v4m): the diagram
    carries the numbered candidates. it["zoom"] (v4z): a third image enlarging those candidates (blanked with the sheet)."""
    from PIL import Image
    im = VC.load_pil(it["image"])
    ab = it.get("ablate")
    if "crops" not in it:
        return [Image.new("RGB", im.size, "white") if ab == "blank_diagram" else im]
    from crops import crop_sheet, marked_diagram, zoom_sheet
    sheet = crop_sheet(im, it["crops"])
    zoom = [zoom_sheet(im, [it["crops"][d - 1] for d in it["zoom"]], it["zoom"])] if it.get("zoom") else []
    if it.get("marks"):                       # v4m: numbered candidates drawn on the diagram (set-of-marks)
        im = marked_diagram(im, it["crops"])
    if ab in ("blank_sheet", "blank_visual"):
        sheet = Image.new("RGB", sheet.size, "white")
        zoom = [Image.new("RGB", z.size, "white") for z in zoom]
    if ab in ("blank_diagram", "blank_visual"):
        im = Image.new("RGB", im.size, "white")
    return [im, sheet] + zoom


_TF = None


def pil_pixel_values(img, max_tiles=VC.MAX_TILES):
    """VC.hf_pixel_values for an in-memory image (same transform and tiling)."""
    import torch
    global _TF
    from internvl_lora_checkpoint_io import _ensure_internvl_repo_on_path
    _ensure_internvl_repo_on_path()
    from internvl.train.dataset import build_transform, dynamic_preprocess
    if _TF is None:
        _TF = build_transform(is_train=False, input_size=448, pad2square=False, normalize_type="imagenet")
    tiles = dynamic_preprocess(img, min_num=1, max_num=max_tiles, image_size=448, use_thumbnail=True)
    return torch.stack([_TF(t) for t in tiles]).to(torch.bfloat16)


def hf_pixels(it: Dict):
    """(pixel values of all the item's images, concatenated in prompt order; tiles per image)."""
    import torch
    if "crops" not in it:
        pv = VC.hf_pixel_values(it["image"])
        return pv, [pv.shape[0]]
    pvs = [pil_pixel_values(im) for im in item_images(it)]
    return torch.cat(pvs), [p.shape[0] for p in pvs]


def hf_prompt_ids(it: Dict, tok, n_tiles: Sequence[int]) -> List[int]:
    """Prompt ids with each <image> expanded in order; must equal what vLLM conditioned on."""
    p = VC.prompt_text(it, "grounding")
    for n in n_tiles:
        p = VC.expand_image(p, n)
    return tok(p).input_ids


class GCEngine(VB.DemandEngine):
    def generate(self, items: List[Dict], n, temperature, max_tokens: Sequence[int], ngram,
                 logprobs=False) -> List[Dict]:
        """Same record per item as DemandEngine.main. prompt_text reads item["system"] and item["user"],
        so grounder and completer prompts go through one path. A v3 completer item also sends its crop sheet
        (the engine needs limit_images=2)."""
        from rollout import completion_ids
        imgs = list(self.eng._io.map(item_images, items))
        prompts = [{"prompt": VC.prompt_text(it, "grounding"), "multi_modal_data": {"image": im[0] if len(im) == 1 else im}}
                   for it, im in zip(items, imgs)]
        sps = [self._sp(n, temperature, 1.0, m, ngram, logprobs) for m in max_tokens]
        res = []
        for o in self.eng.llm.generate(prompts, sps, lora_request=self.eng.lora, use_tqdm=False):
            comps = []
            for c in o.outputs:
                ids = completion_ids(c.token_ids, c.finish_reason, self.eos)
                lps = None
                if logprobs and c.logprobs is not None:
                    lps = [c.logprobs[j][t].logprob for j, t in enumerate(c.token_ids)]
                comps.append({"ids": ids, "trunc": self.eos not in ids, "lps": lps})
            res.append({"prompt_ids": list(o.prompt_token_ids), "comps": comps})
        return res


def fresh_vllm_lora(peft_dir, name: str) -> Path:
    """peft_dir_to_vllm, but never from its /dev/shm cache: a rerun into the same output dir would
    silently serve the previous run's adapter. The pid keeps seeds on different GPUs apart. Caller removes the directory."""
    out = VC.MERGED_ROOT / "lora_cache" / f"gc_{os.getpid()}_{name}"
    shutil.rmtree(out, ignore_errors=True)
    return VC.peft_dir_to_vllm(peft_dir, out)


# torch.distributed / vLLM variables a colocated engine may have set in the trainer. Inherited by the worker,
# they can point its engine at the trainer's rendezvous port -- one candidate cause of the init hang.
_DIST_ENV = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
             "ROLE_RANK", "TORCHELASTIC_RUN_ID", "VLLM_PORT")


class RemoteCompleter:
    """The frozen completer, in its own process (completer_worker.py). Same generate() as GCEngine, for
    greedy calls.

    gpu= pins the worker to its own CUDA device (no memory contention with the trainer's engine and the HF
    policy); the worker gets a clean distributed environment; it runs in its own process group so close()
    can take vLLM's EngineCore child down with it; close() is idempotent and registered at exit."""

    def __init__(self, base, gpu_mem, seed=17, max_model_len=7168, max_num_seqs=64, timeout=900, gpu=None,
                 limit_images=1):
        import atexit
        import queue
        import subprocess
        import threading
        self.timeout, self._closed = timeout, False
        env = {k: v for k, v in os.environ.items() if k not in _DIST_ENV}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.p = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("completer_worker.py")), "--base", str(base),
             "--gpu-mem", str(gpu_mem), "--seed", str(seed), "--max-model-len", str(max_model_len),
             "--max-num-seqs", str(max_num_seqs), "--limit-images", str(limit_images)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            env=env, start_new_session=True)
        atexit.register(self.close)
        self.q = queue.Queue()

        def pump():
            for line in self.p.stdout:
                if line.startswith("@@"):
                    self.q.put(line[2:])
                else:
                    sys.stderr.write(line)
            self.q.put(None)
        threading.Thread(target=pump, daemon=True).start()
        self._read()

    def _read(self):
        import json
        import queue
        try:
            line = self.q.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError(f"completer worker silent for {self.timeout} s")
        if line is None:
            raise RuntimeError(f"completer worker died, exit {self.p.poll()}")
        return json.loads(line)

    def _call(self, msg):
        import json
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        return self._read()

    def set_lora(self, path):
        self._call({"lora": None if path is None else str(path)})

    def generate(self, items, n, temperature, max_tokens, ngram):
        keep = ("image", "system", "user", "crops", "ablate", "marks")
        res = self._call({"items": [{k: it[k] for k in keep if k in it} for it in items], "n": n, "temperature": temperature,
                          "max_tokens": list(max_tokens), "ngram": ngram})
        return res["gens"]

    def close(self):
        """EOF on stdin lets the worker exit normally and shut its engine down. Then signal the whole process
        group: killing only the worker orphans vLLM's EngineCore, which keeps its GPU memory."""
        import signal
        import subprocess
        if self._closed:
            return
        self._closed = True
        try:
            self.p.stdin.close()
        except OSError:
            pass
        try:
            self.p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            pass
        # the worker's own session: reap anything it left behind (vLLM's EngineCore), even after a clean exit
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(self.p.pid, sig)
            except ProcessLookupError:
                return                                   # group already empty
            try:
                self.p.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(0.5)
