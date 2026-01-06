# 🫁 Lung Cancer Detection

Build a powerful image classifier using Convolutional Neural Networks (CNNs) to distinguish between normal lung tissues and cancerous tissues from histopathology images.

---

## 🚀 Project Overview

This repository contains code and an interactive web app for the detection and classification of lung cancer subtypes:
- **Normal**
- **Lung Adenocarcinoma**
- **Lung Squamous Cell Carcinoma**

CNNs are leveraged to automatically extract features and make predictions on medical images.

---

## 📂 Dataset

- **Source:** [Kaggle - Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

To download directly in code:
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/lung-and-colon-cancer-histopathological-images")
print("Path to dataset files:", path)
```

**Dataset Structure:**  
- `lung_aca`: Lung Adenocarcinoma
- `lung_n`: Normal lung tissue
- `lung_scc`: Lung Squamous Cell Carcinoma

---

## 🏗️ Features

- CNN-based architecture for robust image classification
- Train/validate/test pipeline
- Model evaluation with accuracy, precision, recall, and F1-score
- Interactive and stylish [Streamlit](https://streamlit.io/) web app (`app.py`) for real-time predictions
- Data visualization and sample images
- Plots of training and validation accuracy

---

## 📦 Installation & Usage

1. **Clone this repo:**
   ```bash
   git clone https://github.com/aravinds-py/Lung-Cancer-Detection.git
   cd Lung-Cancer-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or install individually:
   pip install tensorflow streamlit pillow kaggle kagglehub pandas matplotlib scikit-learn
   ```

3. **Download the dataset:**
   - Via Kaggle CLI, or use the code above.
   - Place images as required by `Lung Cancer.ipynb`.

4. **Train the model (Jupyter Notebook):**
   - Open and run `Lung Cancer.ipynb`.
   - You can visualize samples and train your CNN using the notebook.

5. **Run the Streamlit app:**
   - Make sure you have the trained model file (`lung_classifier.keras`) in the repo folder.
   - Then run:
     ```bash
     streamlit run app.py
     ```
   - Visit the local URL provided to use the prediction web UI.


---

## 🎨 App Demo

The deployed app provides:

- User-friendly upload for lung histopathology images
- Stylish results with confidence scores and prediction bars

> **For demonstration and research purposes. Not for clinical use.**

---

## 📁 File Structure

- `Lung Cancer.ipynb` – Complete notebook for EDA, preprocessing, training, and evaluation
- `app.py` – Streamlit web app for interactive predictions
- `lung_classifier.keras` – Saved trained model (IPYNB needs to be run to generate)
- `Lung-cancer-detection.jpg` – Project banner/sample image
- `healthcare.png` – UI icon/banner for app
- `.gitignore`, `.gitattributes` – Standard repo boilerplate

---

## 🤝 Contribution & License

Feel free to fork, improve, or open issues for suggestions!

---

## 📣 Acknowledgements

- Data: [Andrew Mvd - Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- Streamlit for interactive ML deployment

---

*Made with ❤️ by [aravinds-py](https://github.com/aravinds-py)*