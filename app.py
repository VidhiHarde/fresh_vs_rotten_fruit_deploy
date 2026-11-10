import json, time
import streamlit as st
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps, ImageDraw, ImageFont

from pathlib import Path
BASE = Path(__file__).parent

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False

st.set_page_config(page_title="Fresh vs Rotten", layout="centered")

st.markdown("""
<style>
    ::selection { text-align: center; background: #10b981; color: #030A30; }

    .stApp {
        background-color: #090B1F;
    }
    /* Center everything inside the main app container */
    .block-container {
        padding-top: 3rem;
        max-width: 700px;
        background: linear-gradient( #1F090B 100%);
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .block-container > div {
        width: 100%;
    }
    
    /* Vibrant Title */
    .title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #C2DFED, #A5AFE4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    /* Clean Subtitle */
    .subtitle {
        color: #EBEEF9;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Centered Result Cards */
    .card {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.2rem auto;
        border-left: 8px solid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        width: 90%;
        text-align: center;
    }
    
    /* Fresh State */
    .fresh {
        background: #ecfdf5;
        border-color: #10b981;
        color: #065f46;
    }
    
    /* Rotten State */
    .rotten {
        background: #fef2f2;
        border-color: #ef4444;
        color: #991b1b;
    }
    
    .row { font-size: 1.3rem; margin: 0.4rem 0; text-align: center; }
    .label { font-weight: 500; opacity: 0.85; }
    .value { font-weight: 700; }

    /* Center file uploader & images */
    .stFileUploader {
        max-width: 400px;
        margin: 0 auto;
    }
    
    div.stImage img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">Fresh vs Rotten Fruit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check the freshness of your fruits</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Apple | Banana | Orange | Mango</div>', unsafe_allow_html=True)


class FruitNet(nn.Module):
    def __init__(self, n_fruit, n_fresh, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.features = models.mobilenet_v2(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fruit_head = nn.Linear(1280, n_fruit)
        self.fresh_head = nn.Linear(1280, n_fresh)
    def forward(self, x):
        x = self.drop(self.pool(self.features(x)).flatten(1))
        return self.fruit_head(x), self.fresh_head(x)

@st.cache_resource
def load_model():
    labels = json.load(open(BASE / "labels.json"))
    model = FruitNet(len(labels["fruits"]), len(labels["fresh"]), pretrained=False)
    model.load_state_dict(torch.load(BASE / "fruit_model.pth", map_location="cpu"))
    model.eval()
    return model, labels["fruits"], labels["fresh"]

model, fruits, fresh_names = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image):
    image = ImageOps.fit(image, (224, 224))
    x = transform(image).unsqueeze(0)
    start = time.time()
    with torch.no_grad():
        out_fruit, out_fresh = model(x)
        p_fruit = torch.softmax(out_fruit, 1)[0]
        p_fresh = torch.softmax(out_fresh, 1)[0]
    ms = (time.time() - start) * 1000
    fc, fi = p_fruit.max(0)
    sc, si = p_fresh.max(0)
    return fruits[fi.item()], fc.item(), fresh_names[si.item()], sc.item(), ms

def show_result(image):
    fruit, fc, status, sc, ms = predict(image)
    st.markdown(f"""
    <div class="card {status}">
        <div class="row"><span class="label">Fruit : </span><span class="value">{fruit.title()}</span></div>
        <div class="row"><span class="label">Status : </span><span class="value">{status.title()}</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"**Fruit confidence:** {fc * 100:.1f}%")
    st.progress(fc)
    st.markdown(f"**Status confidence:** {sc * 100:.1f}%")
    st.progress(sc)
    if min(fc, sc) < 0.60:
        st.warning("Low confidence. Place one fruit in the centre with good lighting and retry.")
    st.image(image, caption="Input image", width=350)
    st.caption(f"Inference time: {ms:.1f} ms")

def load_image(file):
    return ImageOps.exif_transpose(Image.open(file)).convert("RGB")

if WEBRTC_OK:
    class FruitProcessor(VideoProcessorBase):
        def __init__(self):
            self.count = 0
            self.fruit, self.status, self.fc, self.sc = "", "", 0.0, 0.0

        def recv(self, frame):
            img = frame.to_image().convert("RGB")
            self.count += 1
            if self.count % 3 == 1:
                self.fruit, self.fc, self.status, self.sc, _ = predict(img)
            colour = (16, 185, 129) if self.status == "fresh" else (239, 68, 68)
            text = f"{self.fruit.title()} {self.fc*100:.0f}% | {self.status.title()} {self.sc*100:.0f}%"
            if min(self.fc, self.sc) < 0.60:
                colour, text = (120, 120, 120), "Not sure"
            draw = ImageDraw.Draw(img)
            w, h = img.size
            try:
                font = ImageFont.load_default(size=max(18, w // 24))
            except TypeError:
                font = ImageFont.load_default()
            draw.rectangle([0, 0, w - 1, h - 1], outline=colour, width=6)
            draw.rectangle([0, 0, w, max(40, w // 12)], fill=colour)
            draw.text((12, 6), text, fill=(255, 255, 255), font=font)
            return av.VideoFrame.from_image(img)

tab1, tab2 = st.tabs(["Upload", "Camera"])

with tab1:
    file = st.file_uploader("Upload a fruit photo", type=["jpg", "jpeg", "png"])
    if file is not None:
        show_result(load_image(file))

with tab2:
    photo = st.camera_input("Take a photo of the fruit")
    if photo is not None:
        show_result(load_image(photo))
