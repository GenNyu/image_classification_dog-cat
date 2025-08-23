import os, sys, re, tempfile, subprocess
import streamlit as st
from PIL import Image

# ---- cấu hình mặc định ----
SELF_SCRIPT = "predict.py"
SELF_MODEL  = "models/dogcat_mobilenetv2.keras"

TM_SCRIPT   = "predict_tm.py"
TM_MODEL    = "models/keras_model.h5"
TM_LABELS   = "models/labels.txt"  

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐱🐶", layout="centered")
st.title("🐶🐱 Dog vs Cat Classifier")
st.caption("Chọn chế độ • Upload 1 ảnh • Nhận kết quả")

# ---------- UI ----------
mode = st.radio("Chọn chế độ", ["Tự train", "Teachable Machine"], index=0, horizontal=True)
uploaded = st.file_uploader("Chọn 1 ảnh (PNG/JPG/WEBP)", type=["png","jpg","jpeg","webp"])
run = st.button("Predict", type="primary", use_container_width=True)

_NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def _parse_output_tm(text: str):
    label, conf = None, None
    for line in text.splitlines():
        t = line.strip()
        low = t.lower()
        if low.startswith("class:"):
            label = t.split(":", 1)[1].strip()
        elif "confidence" in low:
            m = re.search(_NUM_RE, t)
            if m: conf = float(m.group(0))
    if label is None or conf is None:
        m = re.search(r"([A-Za-z][\w\-]*)\s*\(\s*(" + _NUM_RE + r")\s*\)", text)
        if m: return m.group(1), float(m.group(2))
    return label, conf

def _parse_output_self(text: str):
    m = re.search(r"([A-Za-z][\w\-]*)\s*\(\s*(" + _NUM_RE + r")\s*\)", text)
    if m:
        return m.group(1), float(m.group(2))
    return _parse_output_tm(text)

def _check_files(mode):
    if mode == "Tự train":
        needed = (SELF_SCRIPT, SELF_MODEL)
    else:
        needed = (TM_SCRIPT, TM_MODEL, TM_LABELS)  
    return [p for p in needed if not os.path.isfile(p)]

if run:
    if not uploaded:
        st.error("Vui lòng upload 1 ảnh.")
    else:
        missing = _check_files(mode)
        if missing:
            st.error("Thiếu file: " + ", ".join(missing))
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    img = Image.open(uploaded).convert("RGB")
                    img.save(tmp.name, format="PNG")
                    tmp_path = tmp.name

                if mode == "Tự train":
                    cmd = [sys.executable, SELF_SCRIPT, tmp_path, "--model", SELF_MODEL]
                else:
                    cmd = [sys.executable, TM_SCRIPT, tmp_path, "--model", TM_MODEL, "--labels", TM_LABELS]

                with st.spinner("Đang dự đoán…"):
                    proc = subprocess.run(cmd, capture_output=True, text=True, shell=False)

                to_parse = proc.stdout if (proc.stdout and proc.stdout.strip()) else \
                           ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))

                label, conf = (_parse_output_self(to_parse) if mode=="Tự train" else _parse_output_tm(to_parse))

                st.image(img, caption="Ảnh đầu vào", use_container_width=True)

                if proc.returncode != 0:
                    st.error("Script trả về lỗi. Xem log bên dưới:")
                    st.code((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))
                elif label is not None and conf is not None:
                    st.success(f"Kết quả: **{label}** — Độ tự tin: **{conf:.3f}**")
                else:
                    st.warning("Không parse được kết quả, hiển thị log:")
                    st.code((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))

            except Exception as e:
                st.exception(e)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except: pass
