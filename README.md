# 🎬 Real-Time Face Segmentation for Movie Scene Cast Identification

## 📌 Overview

This project builds a **real-time face segmentation system** for movie scenes, enabling users to detect faces and identify actors directly from images or video frames.

It compares two deep learning approaches:

* **U-Net (CNN-based)**
* **SegFormer (Transformer-based)**

The system is deployed using **Streamlit** for an interactive user experience.

---

## 🚀 Key Features

* 🎯 Real-time face segmentation (<100 ms inference)
* 🧠 U-Net with MobileNetV2 (Transfer Learning)
* 🤖 SegFormer (Transformer-based segmentation)
* 📊 Model comparison (Accuracy vs Speed)
* 🎥 Streamlit web app (image + webcam support)
* 📁 Automated dataset creation (celebrity image downloader)
* 💾 Model saving, loading & verification

---

## 🧠 Models Used

### 🔹 1. U-Net (CNN-Based)

* Backbone: **MobileNetV2**
* Loss Function: **Dice Loss + Binary Crossentropy**
* Dice Score: **~0.88**
* Strength: Works well on small datasets

---

### 🔹 2. SegFormer (Transformer-Based)

* Model: `nvidia/mit-b0`
* Faster inference (~14 ms)
* Better global context understanding
* Limitation: Needs more data for higher accuracy

---

## 📊 Model Performance Comparison

| Metric           | U-Net | SegFormer | Target  |
| ---------------- | ----- | --------- | ------- |
| Dice Coefficient | 0.88  | 0.87      | >0.92   |
| IoU Score        | 0.79  | 0.77      | >0.88   |
| F1 Score         | 0.88  | 0.87      | >0.90   |
| Inference Speed  | 75 ms | **14 ms** | <100 ms |

---

## ⚠️ Key Insight

Although SegFormer is a more advanced architecture, **U-Net outperformed it in accuracy** due to:

* Small dataset size (~400 images)
* Use of **bounding box–derived masks** instead of pixel-level segmentation

👉 This highlights an important ML lesson:

> **Better models don’t guarantee better performance — data quality matters more.**

---

## 🛠️ Tech Stack

* **Languages:** Python
* **Libraries:**

  * TensorFlow / Keras
  * PyTorch
  * Hugging Face Transformers
  * OpenCV
  * NumPy, Matplotlib, Scikit-learn
* **Deployment:** Streamlit

---

## 📂 Project Structure

```bash
Face-Segmentation-Movie-Scene/
│
├── app/                     # Streamlit app
│   └── app.py
│
├── notebook/               # Training notebooks
│
├── models/                 # Saved models
│   ├── unet_face_segmentation.keras
│   └── segformer_face/
│
├── celebrity_faces/        # Auto-downloaded images (generated)
│
├── download_celebrities.py # Script to download dataset
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

Instead of storing large images in the repository, this project uses a script to download sample celebrity faces.

Run:

```bash
python download_celebrities.py
```

This will create:

```bash
celebrity_faces/
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📸 Demo

👉 Add screenshots here (recommended)

Example:

* Input image
* Ground truth mask
* Predicted mask

---

## 🧪 Evaluation Metrics

* Dice Coefficient
* IoU (Intersection over Union)
* F1 Score
* Inference Speed (ms/image)

---

## 🔍 Key Learnings

* Transfer learning significantly improves performance
* Data quality is more important than model complexity
* CNNs can outperform Transformers on small datasets
* Real-time constraints require model optimization
* Proper evaluation and visualization are critical

---

## 🚀 Future Improvements

* Use pixel-level segmentation dataset (CelebAMask-HQ)
* Improve mask quality (polygon-based annotations)
* Add face recognition (name prediction)
* Deploy on cloud (Hugging Face / AWS)
* Optimize for mobile inference

---

## 👨‍💻 Author

**Rudra Barman**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!
