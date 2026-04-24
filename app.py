import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU mode

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import time
import io

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title='Scene Cast AI', page_icon='🎬', layout='wide')

st.markdown("""
<style>
    .main-title { text-align: center; color: #FF4B4B; font-size: 2.5rem; font-weight: bold; }
    .actor-card {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        border: 1px solid #FF4B4B;
    }
    .known-card {
        background: #1E2E1E;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        border: 2px solid #00FF00;
    }
    .stButton>button { background-color: #FF4B4B; color: white; border-radius: 8px; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── DeepFace Import ───────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "unet_face_segmentation.keras")
FACES_DIR  = os.path.join(BASE_DIR, "celebrity_faces")

# ── Custom Loss Functions ─────────────────────────────────────
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)

# ── Load Model (Cached) ───────────────────────────────────────
@st.cache_resource
def load_unet():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'dice_coefficient': dice_coefficient,
            'dice_loss'       : dice_loss,
            'combined_loss'   : combined_loss
        }
    )

# ── Face Segmentation ─────────────────────────────────────────
def predict_mask(model, image, threshold=0.5):
    img = np.array(image.convert('RGB'))
    h, w = img.shape[:2]
    inp   = np.expand_dims(cv2.resize(img, (256, 256)).astype(np.float32) / 255., 0)
    t     = time.time()
    pred  = model.predict(inp, verbose=0)
    speed = (time.time() - t) * 1000
    mask256 = (pred[0, :, :, 0] > threshold).astype(np.uint8)
    mask    = cv2.resize(mask256, (w, h), interpolation=cv2.INTER_NEAREST)
    conf    = float(pred[0, :, :, 0].max())
    overlay = img.copy()
    overlay[mask == 1] = (
        overlay[mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    n, _ = cv2.connectedComponents(mask)
    return mask, overlay, speed, conf, n - 1, img

# ── Celebrity Recognition ─────────────────────────────────────
def recognize_celebrities(img_rgb, mask):
    results = []
    if not DEEPFACE_AVAILABLE:
        return results

    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    for lid in range(1, n_labels):
        region = (labels == lid).astype(np.uint8)
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue

        x1 = max(0, xs.min() - 20);  x2 = min(img_rgb.shape[1], xs.max() + 20)
        y1 = max(0, ys.min() - 20);  y2 = min(img_rgb.shape[0],  ys.max() + 20)
        crop = img_rgb[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            continue

        actor_name = 'Unknown'
        age = gender = emotion = '?'
        distance_val = 999.0

        try:
            tmp = os.path.join(BASE_DIR, 'tmp_face.jpg')
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp, crop_bgr)

            if os.path.exists(FACES_DIR):
                face_files = [f for f in os.listdir(FACES_DIR)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(face_files) > 0:
                    for model_name in ['VGG-Face', 'Facenet']:
                        try:
                            df_res = DeepFace.find(
                                img_path=tmp,
                                db_path=FACES_DIR,
                                model_name=model_name,
                                distance_metric='cosine',
                                threshold=0.6,
                                enforce_detection=False,
                                silent=True
                            )
                            if len(df_res) > 0 and len(df_res[0]) > 0:
                                best_row = df_res[0].iloc[0]
                                dist     = best_row.get('distance', 999.0)
                                if dist < distance_val:
                                    distance_val = dist
                                    best_path    = best_row['identity']
                                    fname = os.path.splitext(
                                        os.path.basename(best_path)
                                    )[0]
                                    actor_name = fname.replace('_', ' ')
                            if distance_val < 0.6:
                                break
                        except Exception:
                            continue

                    if distance_val >= 0.6:
                        actor_name = 'Unknown'

        except Exception:
            actor_name = 'Unknown'

        try:
            ana = DeepFace.analyze(
                crop,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                silent=True
            )
            if isinstance(ana, list):
                ana = ana[0]
            age     = ana.get('age', '?')
            gender  = ana.get('dominant_gender', '?')
            emotion = ana.get('dominant_emotion', '?')
        except Exception:
            pass

        results.append({
            'face_id' : lid,
            'bbox'    : (x1, y1, x2, y2),
            'name'    : actor_name,
            'age'     : age,
            'gender'  : gender,
            'emotion' : emotion,
            'distance': round(distance_val, 3)
        })

    return results

# ── Draw Bounding Boxes ───────────────────────────────────────
def draw_boxes(img_rgb, results):
    out = img_rgb.copy()
    for r in results:
        x1, y1, x2, y2 = r['bbox']
        name  = r['name']
        color = (0, 255, 0) if name != 'Unknown' else (0, 165, 255)
        label = f"{name} | Age:{r['age']} | {r['emotion']}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out

def to_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()

# ══════════════════════ MAIN APP ═════════════════════════════
st.markdown('<h1 class="main-title">🎬 Scene Cast AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:gray;">Real-Time Face Segmentation + Celebrity Recognition</p>',
    unsafe_allow_html=True
)
st.markdown('---')

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title('⚙️ Settings')
    threshold    = st.slider('Detection Threshold', 0.1, 0.9, 0.5, 0.05)
    show_overlay = st.checkbox('Show Green Overlay',      value=True)
    show_mask    = st.checkbox('Show Binary Mask',        value=True)
    show_celeb   = st.checkbox('Show Celebrity Analysis', value=True)
    st.markdown('---')
    st.markdown('### 📊 Model Info')
    st.info('**Model:** U-Net + MobileNetV2')
    st.info('**Val Dice:** 0.8826')
    c1, c2 = st.columns(2)
    c1.metric('Dice',  '0.885'); c2.metric('IoU',  '0.794')
    c1.metric('F1',    '0.885'); c2.metric('Speed', '75ms')
    st.markdown('---')
    st.markdown('**🌟 Actor Database:**')
    st.markdown("""
    - Scarlett Johansson
    - Chris Hemsworth
    - Robert Downey Jr
    - Chris Evans
    - Tom Holland
    - Zendaya
    - Benedict Cumberbatch
    - Josh Brolin
    - Dwayne Johnson
    - Ram Charan
    - N T Rama Rao Jr
    """)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['📸 Image Upload', '📊 Dashboard', 'ℹ️ About'])

# ════════ TAB 1 ════════
with tab1:
    st.subheader('Upload a Movie Scene Screenshot')
    uploaded = st.file_uploader('Choose image...', type=['jpg', 'jpeg', 'png'])

    if uploaded:
        image = Image.open(uploaded)

        # ✅ LAZY LOAD — model sirf tab load hoga jab image upload ho
        with st.spinner('🔄 Loading model... (first time ~60s)'):
            model = load_unet()

        if model is None:
            st.error(f"❌ Model not found at: {MODEL_PATH}")
            st.stop()

        with st.spinner('🔍 Segmenting faces...'):
            mask, overlay, speed, conf, faces, img_rgb = predict_mask(
                model, image, threshold
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('⏱️ Speed',      f'{speed:.0f}ms',
                  delta='✅ Fast' if speed < 100 else '⚠️ Slow')
        m2.metric('🎯 Confidence', f'{conf:.1%}')
        m3.metric('👤 Faces',      faces)
        m4.metric('📐 Size',       f'{image.size[0]}×{image.size[1]}')
        st.markdown('---')

        cols = st.columns(3)
        with cols[0]:
            st.markdown('**Original Image**')
            st.image(image, use_container_width=True)
        if show_overlay:
            with cols[1]:
                st.markdown('**Face Overlay 🟢**')
                st.image(overlay, use_container_width=True)
        if show_mask:
            with cols[2]:
                st.markdown('**Binary Mask**')
                st.image(mask * 255, use_container_width=True, clamp=True)

        st.markdown('---')

        if show_celeb and faces > 0:
            st.subheader('🎭 Celebrity Recognition')
            if not DEEPFACE_AVAILABLE:
                st.warning('DeepFace not installed.')
            else:
                with st.spinner('🔍 Identifying actors...'):
                    results = recognize_celebrities(img_rgb, mask)

                if results:
                    st.image(draw_boxes(img_rgb, results),
                             use_container_width=True, caption='Detected Actors')
                    st.markdown('### 👥 Cast Details')
                    cols2 = st.columns(min(len(results), 4))
                    for i, r in enumerate(results):
                        with cols2[i % 4]:
                            is_known = r['name'] != 'Unknown'
                            icon     = '🌟' if is_known else '👤'
                            card_cls = 'known-card' if is_known else 'actor-card'
                            dist_txt = f"Match: {r['distance']}" if is_known else ''
                            st.markdown(
                                f'<div class="{card_cls}">'
                                f'<h3>{icon} {r["name"]}</h3>'
                                f'<p>🎂 Age: <b>{r["age"]}</b></p>'
                                f'<p>⚧ Gender: <b>{r["gender"]}</b></p>'
                                f'<p>😊 Emotion: <b>{r["emotion"]}</b></p>'
                                f'<p style="font-size:0.8rem;color:gray;">{dist_txt}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.warning('No faces could be analyzed.')

        st.markdown('---')
        st.subheader('📥 Download Results')
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button('⬇️ Download Mask',
                to_bytes((mask * 255).astype(np.uint8)), 'face_mask.png', 'image/png')
        with d2:
            st.download_button('⬇️ Download Overlay',
                to_bytes(overlay), 'face_overlay.png', 'image/png')
        with d3:
            log = f"""Scene Cast AI — Detection Log
==============================
File      : {uploaded.name}
Faces     : {faces}
Confidence: {conf:.1%}
Speed     : {speed:.0f}ms
Threshold : {threshold}
Size      : {image.size[0]}x{image.size[1]}
Model     : U-Net + MobileNetV2
"""
            st.download_button('⬇️ Download Log', log, 'detection_log.txt', 'text/plain')

    else:
        st.info('👆 Upload a movie scene screenshot to detect and identify actors!')
        st.markdown("""
        ### 📌 How to use:
        1. Upload any movie scene screenshot (JPG / PNG)
        2. Model will detect and segment faces automatically
        3. Celebrity Recognition will identify known actors
        4. Download results — mask, overlay, detection log

        ### 💡 For better recognition:
        - Add multiple photos of same actor in `celebrity_faces` folder
        - Use clear front-facing photos
        - Higher resolution images work better
        """)

# ════════ TAB 2 ════════
with tab2:
    st.subheader('📊 Model Performance Dashboard')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Dice Coefficient', '0.8853', delta='+0.8853 vs baseline')
    c2.metric('IoU Score',        '0.7943')
    c3.metric('F1 Score',         '0.8853')
    c4.metric('Inference Speed',  '75.9ms', delta='-24ms vs target')
    st.markdown('---')
    st.subheader('📈 Model Comparison')
    import pandas as pd
    df = pd.DataFrame({
        'Metric'   : ['Dice Coefficient', 'IoU Score', 'F1 Score', 'Speed (ms)'],
        'U-Net'    : [0.8853, 0.7943, 0.8853, 75.9],
        'SegFormer': [0.8631, 0.7591, 0.8631, 10.6],
        'Target'   : [0.92,   0.88,   0.90,  100.0],
    })
    st.dataframe(df, use_container_width=True)
    st.markdown('---')
    st.subheader('🏋️ Training Summary')
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        **U-Net Training:**
        - Total Samples : 1,636 (409 × 4 augmentations)
        - Train/Val/Test: 1308 / 164 / 164
        - Epochs        : 49 (early stopping)
        - Best Val Dice : **0.8826**
        - Optimizer     : Adam (lr: 1e-3 → 8e-6)
        """)
    with t2:
        st.markdown("""
        **SegFormer Training:**
        - Base Model : nvidia/mit-b0
        - Epochs     : 10
        - Final Loss : 0.0724
        - Val Dice   : **0.8631**
        - Speed      : **10.6ms** (7x faster)
        """)

# ════════ TAB 3 ════════
with tab3:
    st.subheader('ℹ️ About This Project')
    st.markdown("""
    ### 🎬 Scene Cast AI — Real-Time Face Segmentation

    This app automatically detects and segments faces in movie scene screenshots,
    enabling users to pause videos and instantly view cast/crew details.

    **Tech Stack:**
    - 🧠 Model      : U-Net + MobileNetV2 (Transfer Learning)
    - 🔥 Framework  : TensorFlow / Keras
    - 🤗 Model 2    : SegFormer (HuggingFace nvidia/mit-b0)
    - 👁️ Recognition : DeepFace VGG-Face + Facenet
    - 🌐 App        : Streamlit
    - 📦 Libraries  : OpenCV, NumPy, scikit-learn, Pandas

    **Business Value:**
    - ⏸️ Pause-and-identify actors in real time
    - 🎯 Personalized recommendations based on actors
    - 🔞 Content moderation via face detection
    - 📢 Dynamic ads featuring favourite actors

    **Dataset:** 409 movie scene images with bbox face annotations

    **Celebrity Database:** 11 actors pre-loaded
    (Scarlett Johansson, Chris Hemsworth, Robert Downey Jr,
    Chris Evans, Tom Holland, Zendaya, Benedict Cumberbatch,
    Josh Brolin, Dwayne Johnson, Ram Charan, N T Rama Rao Jr)
    """)
