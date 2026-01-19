from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import open_clip

app = FastAPI(title="Local CLIP Classifier", version="1.0.0")

# ---------
# 1) Завантажуємо модель ОДИН раз при старті
# ---------
DEVICE = "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    device=DEVICE,
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# Тексти-класи (їх можна потім покращувати, але це працює "з коробки")
TEXTS_AD = [
    "a screenshot of an advertisement",
    "an ad creative screenshot",
    "a banner ad screenshot",
]
TEXTS_LANDING = [
    "a screenshot of a landing page",
    "a screenshot of a website page",
    "a screenshot of an app store page",
    "a screenshot of a Google Play page",
]

@torch.inference_mode()
def classify_image(pil_image: Image.Image):
    # preprocess -> tensor [1, 3, H, W]
    image_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    # Робимо 2 логіти: ad vs landing
    texts = TEXTS_AD + TEXTS_LANDING
    text_tokens = tokenizer(texts).to(DEVICE)

    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tokens)

    # Нормалізація (стандарт для CLIP)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Схожість
    logits = (image_features @ text_features.T).squeeze(0)  # shape: [len(texts)]

    # Беремо найкращий score для ad і landing
    ad_logits = logits[:len(TEXTS_AD)]
    landing_logits = logits[len(TEXTS_AD):]

    best_ad = torch.max(ad_logits)
    best_landing = torch.max(landing_logits)

    # Перетворимо в "confidence" через softmax на 2 класи
    two = torch.stack([best_ad, best_landing], dim=0)
    probs = torch.softmax(two, dim=0)

    ad_prob = float(probs[0].item())
    landing_prob = float(probs[1].item())

    if ad_prob >= landing_prob:
        return "ad", ad_prob
    else:
        return "landing", landing_prob

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Перевірка типу
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "Please upload an image file (content-type must start with image/)."}

    # Читаємо байти
    raw = await file.read()

    # Відкриваємо як PIL Image
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return {"error": "Cannot read this file as an image."}

    label, conf = classify_image(pil)

    return {
        "type": label,
        "confidence": round(conf, 4),
    }
