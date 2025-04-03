# Deep Learning Approach for Improvement in Melanoma Diagnosis for Dark-Skinned Patients

**Team Members:** Savannah Chan, Hana Kim, Hyun Seo Lee, Isalis Karhu-Leperd, Justin Wong  
**Course:** Machine Learning for Medical Application (EN.520.439/659), Johns Hopkins University  

---

## 🔍 Motivation

Melanoma is a serious form of skin cancer affecting approximately 100,000 people annually in the U.S. While lighter-skinned individuals are more frequently diagnosed, darker-skinned individuals face significantly **higher mortality rates** due to **delayed detection** and **diagnostic disparities**. 

Traditional diagnostic methods like **biopsy** are invasive and slow, while **AI-based tools** are often **biased**, underperforming on darker skin due to **imbalanced training data**.

---

## 📚 Literature Review

- **Bias in AI Models:** Many models are trained on light-skinned images, yielding poor performance for dark-skinned patients.
- **Attempts at Fairness:** Previous work includes dataset creation (e.g., Barros et al., 2023) and synthetic image generation (e.g., Rezk et al., 2022), but limitations remain.
- **Need for Realism:** Synthetic dark-skin images often lack accurate visual and diagnostic characteristics, limiting their effectiveness in training.

---

## 🧪 Methodology

Our approach aims to **balance skin tone representation** in melanoma datasets and **enhance model generalizability** using:

### 🔁 Classical Augmentation
- Color shifting, darkening, Gaussian noise
- Results were unsatisfactory; lesion boundaries were distorted

### 🎨 Deep Learning-Based Style Transfer
- **VGG-19-based style transfer** with input (content) and reference (style) images
- **Google Monk Skin Tone Scale** is used to quantitatively evaluate skin tone
- **Otsu thresholding** removes lesions before skin tone classification

### 🧠 Classification Model
- **EfficientNet-B0**, pre-trained CNN for melanoma classification
- Generated dark-skin images are used for training only; real dark-skin data is used for inference

---

## 📊 Datasets

- **HAM10000:** Primary dataset for augmentation
- **CMB-MEL and ISIC datasets:** Used for classification training and evaluation

#### Skin Tone Distribution (HAM10000)
- Over 98% of samples are fair-skinned
- Only a handful of dark skin tone samples (e.g., 1 medium tone image)

#### Gender & Age Distribution
- Balanced gender split (54% male, 45.5% female)
- Normal distribution of age with a peak at 40–50 years

---

## 📈 Preliminary Results

- Style transfer with lesion removal (Otsu) improves skin tone classification accuracy
- Skin tone imbalance is clearly demonstrated via Monk scale matching
- EfficientNet-B0 model will be fine-tuned on a balanced dataset for melanoma classification

---

## 🚀 Next Steps

1. Use **style transfer** to generate dark-skinned images from light-skinned samples.
2. **Combine synthetic and real data** to create a balanced training set.
3. **Train and evaluate** the EfficientNet-B0 classifier for performance on dark and light skin tones.
4. Evaluate fairness and generalizability using skin tone-aware metrics.

---

## 📌 Significance

This project aims to **bridge diagnostic disparities** for dark-skinned patients by improving AI-based melanoma classification. A generalizable model will contribute to **equitable healthcare access** and **higher survival rates** across all skin tones.

---

## 📚 References

See the full list of references in the [proposal report](./Project_Midterm_Report_MLMA_2025.pdf), including works from Barros (2023), Rezk (2022), and the creators of the HAM10000 and Monk Skin Tone Scale.
