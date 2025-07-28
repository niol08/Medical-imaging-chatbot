import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
from PIL import Image
import pathlib
from utils.FMRI import SimpleCNN
from gemini import query_gemini_rest
import os
from dotenv import load_dotenv
from transformers import AutoImageProcessor, AutoModelForImageClassification




load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

st.set_page_config(page_title="Medical Imaging Chatbot", page_icon="🧠", layout="centered")
st.title("Medical Imaging Chatbot")

tabs = st.tabs(["fMRI Classifier", "X-ray Classifier", "NMR Diagnostic"])
with tabs[0]:
    st.header("fMRI Brain Tumor Classification")

    with st.expander("📄 fMRI Image Requirements"):
        st.markdown(
            "- Upload a `.jpg` or `.png` file of a **2D MRI/fMRI slice**.\n"
            "- Recommended size: at least **224x224** pixels.\n"
            "- Model outputs: Tumor types like Glioma, Meningioma, Pituitary, etc."
        )

        variant = st.radio("Choose model variant", ["f (Fast)", "c (Compact)"])
        variant_key = "f" if variant.startswith("f") else "c"

        @st.cache_resource(show_spinner="🔄 Downloading & loading Vbai‑TS model…")
        def load_fmri_model(v_key: str):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            
            filename = f"Vbai-TS 1.0{v_key}.pt"
            ckpt_file = hf_hub_download(
                repo_id="Neurazum/Vbai-TS-1.0",
                filename=filename,
                cache_dir="hf_cache"        
            )

            state = torch.load(ckpt_file, map_location=device)
            num_classes = state["fc2.weight"].shape[0]

            net = SimpleCNN(model_type=v_key, num_classes=num_classes).to(device)
            net.load_state_dict(state, strict=True)
            net.eval()
            return net, device

        model, device = load_fmri_model(variant_key)

    idx2label = {
        0: "Glioma Tumor",
        1: "Meningioma Tumor",
        2: "Pituitary Tumor",
        3: "No Tumor",
        4: "Class 4 (undocumented)",
        5: "Class 5 (undocumented)"
    }

    uploaded = st.file_uploader("Upload an fMRI or MRI slice (.jpg or .png)", type=["jpg", "jpeg", "png"], key="fmri_upload")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input Slice", use_container_width=True)

        run_check = st.button("Run Check", key="run_check_btn")
        if run_check:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            with torch.inference_mode():
                x = transform(image).unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
                conf, pred = torch.max(probs, dim=0)

            label = idx2label.get(int(pred), f"Class {int(pred)}")
            st.success(f"Prediction ▸ **{label}**")
            st.info(f"Confidence ▸ {conf.item() * 100:.2f}%")

            api_key = os.getenv("GEMINI_API_KEY", "")
            if api_key:
                explanation = query_gemini_rest("fMRI", label, conf.item(), api_key)
                st.markdown("### 🧠 Gemini Insight")
                st.write(explanation)
            else:
                st.info("Gemini API key missing – no explanation generated.")
    else:
        st.info("Upload a file to begin diagnosis.")
        


with tabs[1]:
    st.header("📷 X‑Ray Diagnostic")

    region = st.radio(
        "Region to analyse",
        ["Chest (Pneumonia)", "Limb Fracture"],
        horizontal=True
    )

    if region.startswith("Chest"):
        MODEL_NAME  = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
        LABELS      = ["Normal", "Pneumonia"]
    else:
        MODEL_NAME  = "prithivMLmods/Bone-Fracture-Detection"
        LABELS      = ["Fractured", "Not Fractured"]


    @st.cache_resource(show_spinner="🔄 Loading X‑ray model…")
    def load_xray_assets(name):
        proc  = AutoImageProcessor.from_pretrained(name)
        model = AutoModelForImageClassification.from_pretrained(name)
        model.eval()
        return proc, model

    processor, model = load_xray_assets(MODEL_NAME)

    upl = st.file_uploader("Upload an X‑ray (.jpg / .png)", ["jpg", "jpeg", "png"], key="xray_up")
    if upl and st.button("Run X‑Ray Diagnostic", key="run_xray"):
        img = Image.open(upl).convert("RGB")
        st.image(img, caption="Uploaded X‑ray", use_container_width=True)

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()  

        for i, p in enumerate(probs):
            st.write(f"**{LABELS[i]}** — {p*100:.2f}%")

        
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            best_idx = int(probs.argmax())
            explain  = query_gemini_rest("X‑Ray", LABELS[best_idx], float(probs[best_idx]), api_key)
            st.markdown("### 🧠 Gemini Insight")
            st.write(explain)
        else:
            st.info("Gemini key missing – no explanation generated.")
    else:
        st.info("Upload an image to begin.")


with tabs[2]:
    st.header("🧲 NMR (MRI) Diagnostic")

    MODEL = "prithivMLmods/BrainTumor-Classification-Mini"
    LABELS = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    @st.cache_resource(show_spinner="📦 Loading MRI model…")
    def load_mri_model():
        proc  = AutoImageProcessor.from_pretrained(MODEL)
        mdl   = AutoModelForImageClassification.from_pretrained(MODEL)
        mdl.eval()
        return proc, mdl

    processor, model = load_mri_model()

    upload = st.file_uploader("Upload MRI slice (.jpg/.png)", ["jpg", "jpeg", "png"], key="nmr_up")
    if upload and st.button("Run NMR Diagnostic", key="run_nmr"):
        img = Image.open(upload).convert("RGB")
        st.image(img, caption="Uploaded MRI", use_container_width=True)

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        st.success("### Predictions")
        for i, p in enumerate(probs):
            st.write(f"**{LABELS[i]}** — {p*100:.2f}%")

        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            best = int(probs.argmax())
            explanation = query_gemini_rest("MRI", LABELS[best], float(probs[best]), api_key)
            st.markdown("### 🧠 Gemini Insight")
            st.write(explanation)
        else:
            st.info("Gemini key missing – no explanation generated.")
    else:
        st.info("Upload an image to begin.")
