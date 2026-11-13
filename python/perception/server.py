import argparse
import json
import sys
import time
import os
import io
import base64
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--gpu", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--runtime-config", type=Path, required=True)
parser.add_argument("--backend", choices=["sam3"], required=True)
args = parser.parse_args()


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.runtime.resources import reserve_gpu_memory
root = Path(__file__).resolve().parents[2]
runtime = json.loads((root / args.runtime_config).read_text())
resource_directory = str((root / runtime['resource_domain']['directory']).resolve())
torch.set_num_threads(runtime['cpu']['service_threads'])


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0")
torch.cuda.set_device(device)
reservation = reserve_gpu_memory(device, runtime['perception'], resource_directory, args.gpu, time.time() + runtime['runtime']['gpu_admission_seconds'])
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
del reservation
torch.cuda.empty_cache()
model = build_sam3_image_model(checkpoint_path=args.model, device=str(device), enable_inst_interactivity=True).to(device).eval()
processor = Sam3Processor(model, device=str(device), confidence_threshold=0.0)
app = FastAPI()


class Query(BaseModel):
    image: str
    text: str | None = None
    point: list[int] | None = None
    points: list[list[int]] | None = None
    labels: list[int] | None = None
    box: list[float] | None = None
    min_score: float | None = None


@app.get("/health")
def health():
    return {"backend": args.backend}


@app.post("/segment")
async def segment(query: Query):
    image = Image.open(io.BytesIO(base64.b64decode(query.image))).convert("RGB")
    w, h = image.size
    if sum(x is not None for x in [query.text, query.point, query.points, query.box]) != 1:
        raise HTTPException(422, 'Specify exactly one of text, point, points or box')
    if query.labels is not None and query.points is None:
        raise HTTPException(422, 'labels requires points')
    points = [query.point] if query.point is not None else query.points
    labels = query.labels if query.labels is not None else [1]*len(points) if points is not None else None
    if points is not None and (not points or any(len(p) != 2 or not (0 <= p[0] < h and 0 <= p[1] < w) for p in points) or len(labels) != len(points) or any(v not in (0, 1) for v in labels)):
        raise HTTPException(422, 'points requires in-image [row,col] pairs and one 0/1 label per point')
    if query.box is not None and (len(query.box) != 4 or not all(np.isfinite(query.box)) or not (0 <= query.box[0] < query.box[2] <= w and 0 <= query.box[1] < query.box[3] <= h)):
        raise HTTPException(422, 'box requires in-image [left,top,right,bottom] bounds')
    if query.text is not None and not query.text.strip():
        raise HTTPException(422, 'text must be nonempty')
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        if query.text is not None:
            out = processor.set_text_prompt(prompt=query.text, state=state)
            raw_masks, raw_scores = out["masks"], out["scores"]
        elif points is not None:
            raw_masks, raw_scores, _ = model.predict_inst(state, point_coords=np.array(points, dtype=np.float32)[:, ::-1].copy(), point_labels=np.array(labels), multimask_output=True)
        else:
            raw_masks, raw_scores, _ = model.predict_inst(state, box=np.array(query.box), multimask_output=True)
    masks = [np.asarray(m.cpu() if hasattr(m,"cpu") else m).squeeze() > 0 for m in raw_masks]
    scores = [float(s) for s in raw_scores]
    regions = []
    for mask, score in zip(masks, scores):
        threshold = query.min_score if query.min_score is not None else (0.0 if points is not None else 0.2)
        if score < threshold:
            continue
        rows, cols = np.nonzero(mask)
        if not len(rows):
            continue
        out = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8)*255).save(out, format="PNG")
        regions.append({"mask": base64.b64encode(out.getvalue()).decode(), "score": score, "area": int(mask.sum()), "centroid": [float(rows.mean()),float(cols.mean())], "bbox": [int(cols.min()),int(rows.min()),int(cols.max()),int(rows.max())]})
    return {"found": bool(regions), "backend": args.backend, "regions": regions, "width": w, "height": h}


uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
