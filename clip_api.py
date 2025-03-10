from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image
import torch
import clip
import uuid
import logging

app = FastAPI()

# In-memory store for job results.
job_results = {}

# Request model for incoming image URLs.
class ImageURLs(BaseModel):
    urls: list[str]

# Global variables for the CLIP model, preprocessing, and device.
model = None
preprocess = None
device = None

# Startup event: load the heavy CLIP model.
@app.on_event("startup")
async def startup_event():
    global model, preprocess, device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Loading CLIP model on device: {device} ...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    logging.info("CLIP model loaded successfully.")

# Simple health-check endpoint.
@app.get("/health")
def health():
    return {"status": "ok"}

# Function to process images and rank them.
def process_images(urls):
    # Define text prompts describing a captivating real estate photo.
    prompts = [
        "A highly captivating real estate photo",
        "A beautiful luxury home interior",
        "A photo that grabs attention on Instagram"
    ]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    scored = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                best_score = similarity[0].max().item()
            scored.append((url, best_score))
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")
            scored.append((url, 0.0))
    scored.sort(key=lambda x: x[1], reverse=True)
    top8 = [url for url, score in scored[:8]]
    return top8

# Background task that runs the ranking job.
def run_ranking_job(job_id: str, urls: list[str]):
    result = process_images(urls)
    job_results[job_id] = result

# Endpoint to start ranking images.
@app.post("/start-ranking")
def start_ranking(data: ImageURLs, background_tasks: BackgroundTasks):
    if not data.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_ranking_job, job_id, data.urls)
    return {"job_id": job_id, "status": "processing"}

# Endpoint to retrieve the results for a given job.
@app.get("/job-result/{job_id}")
def get_job_result(job_id: str):
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Job not found or still processing.")
    return {"top8": job_results[job_id]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
