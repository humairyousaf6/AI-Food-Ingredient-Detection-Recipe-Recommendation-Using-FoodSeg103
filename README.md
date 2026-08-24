# 🍽️ AI Food Ingredient Detection & Recipe Recommendation Using FoodSeg103

A deep learning-based computer vision project designed to **detect and segment food ingredients from images and recommend suitable recipes based on the detected ingredients**.

The project uses the **FoodSeg103 dataset** and custom PyTorch semantic segmentation architectures to identify food ingredients at the pixel level. The current implementation focuses on the ingredient detection and segmentation stage, while the recipe recommendation component is designed as the second stage of the overall system.

The complete vision is:

```text
Food Image
     │
     ▼
AI Ingredient Detection
     │
     ▼
Detected Food Ingredients
     │
     ▼
Ingredient-Based Recipe Recommendation
     │
     ▼
Suggested Recipes
```

---

# ✨ Features

* 🍎 Food ingredient detection from images
* 🖼️ Pixel-level semantic segmentation
* 📊 FoodSeg103 dataset
* 🏷️ 104 food ingredient classes
* 📚 Full 4,983 training images
* 🧪 Evaluation on 2,135 images
* 🔍 Supervisely bitmap annotation decoding
* 🧠 Custom PyTorch segmentation architectures
* 🚀 Multiple training and recovery strategies
* 🛡️ Crash-resistant GPU training
* 💾 Automatic model checkpointing
* 📈 IoU-based model evaluation
* 🔄 Training recovery from checkpoints
* ⚡ Google Colab / NVIDIA Tesla T4 optimization
* 🍳 Ingredient-based recipe recommendation — **planned second stage**

---

# 🧠 Project Overview

The project is designed as a two-stage AI system.

### Stage 1 — Food Ingredient Detection

The first stage uses computer vision and semantic segmentation to identify food ingredients from an input image.

```text
Food Image
    │
    ▼
FoodSeg103-trained Model
    │
    ▼
Ingredient Segmentation
    │
    ▼
Detected Ingredients
```

For example, an image containing multiple food items could produce detected ingredients such as:

```text
Tomato
Egg
Onion
Potato
Chicken
```

The current notebook implements this stage using the FoodSeg103 dataset, Supervisely bitmap annotations, PyTorch segmentation models, and IoU-based evaluation.

---

# 🍳 Stage 2 — Recipe Recommendation

Once the ingredients have been detected from the food image, the second stage of the planned system will use those ingredients to recommend suitable recipes.

```text
Detected Ingredients
        │
        ▼
Ingredient Matching
        │
        ▼
Recipe Recommendation Engine
        │
        ▼
Recommended Recipes
```

For example:

```text
Detected Ingredients:
• Chicken
• Tomato
• Onion
• Garlic

        ↓

Possible Recipe Suggestions:
• Chicken Karahi
• Chicken Curry
• Tomato Chicken
• Garlic Chicken
```

The recipe recommendation component can be developed using an ingredient-recipe database, recipe similarity matching, or an AI/LLM-based recommendation system.

### Current Implementation Status

**Stage 1 — Ingredient Detection & Segmentation:** ✅ Implemented

**Stage 2 — Recipe Recommendation:** 🔮 Planned / Future Development

The current `AIRecipeDetector_(1).ipynb` focuses on training and evaluating the food ingredient segmentation model. Recipe recommendation is the next planned component of the project.

---

# 🏗️ Overall Architecture

```text
                   FOOD IMAGE
                       │
                       ▼
             ┌───────────────────┐
             │  Ingredient       │
             │  Detection Model  │
             └─────────┬─────────┘
                       │
                       ▼
             DETECTED INGREDIENTS
                       │
                       ▼
             ┌───────────────────┐
             │ Recipe             │
             │ Recommendation     │
             │ Engine             │
             └─────────┬─────────┘
                       │
                       ▼
                RECOMMENDED
                  RECIPES
```

---

# 📊 Dataset

This project uses the **FoodSeg103** dataset with pixel-level segmentation annotations.

### Dataset Configuration

* **Dataset:** FoodSeg103
* **Training images:** 4,983
* **Evaluation images:** 2,135
* **Segmentation classes:** 104
* **Annotation format:** Supervisely
* **Annotation type:** Bitmap / pixel-level segmentation
* **Training resolution:** Up to 512 × 512 in later experiments

The notebook performs dataset inspection, metadata verification, annotation analysis, bitmap decoding, mask generation, and training preparation before model training.

---

# 🧠 Model & Training

The notebook contains multiple experimental training approaches, including:

* Lightweight custom CNN
* Crash-resistant CNN training
* DeepLabV3+ experimentation
* Custom encoder-decoder segmentation architecture
* Supervisely bitmap decoding
* Full-dataset training
* Gentle recovery training
* Progressive IoU improvement
* Quality-filtered training
* Mixed-precision training experiments
* Overfitting analysis and mitigation

The primary successful approach uses a custom encoder-decoder segmentation architecture trained on FoodSeg103.

---

# 📈 Results

The notebook records the following validation IoU progression:

```text
Initial Training
      │
      ▼
IoU ≈ 0.058
      │
      ▼
Crash-Proof Training
IoU ≈ 0.1114
      │
      ▼
Gentle Recovery
IoU ≈ 0.1363
      │
      ▼
Ultimate Training
IoU ≈ 0.1446  ← Best Recorded
```

### 🏆 Best Recorded Validation IoU

**0.1446**

The best observed result was achieved during the later experimental training phase. Subsequent experiments showed a decline in IoU, demonstrating the effects of overfitting and training instability.

The notebook therefore preserves the best-performing checkpoint rather than simply using the final training epoch.

---

# 💾 Model Checkpoints

Important checkpoints generated during experimentation include:

```text
best_ingredient_model.pth
advanced_ingredient_model_v2.pth
ultimate_ingredient_model_BITMAP_FIXED.pth
ultimate_ingredient_model_CRASH_PROOF.pth
gentle_recovery_model.pth
complete_gentle_recovery_model.pth
ultimate_0_6_iou_model.pth
```

### Best Recovery Model

```text
complete_gentle_recovery_model.pth
```

Best IoU from the gentle recovery stage:

```text
0.1363
```

### Best Recorded Training Checkpoint

```text
ultimate_0_6_iou_model.pth
```

Best recorded validation IoU:

```text
0.1446
```

> **Note:** `0.6` in the checkpoint filename represents the experimental target and was not the achieved IoU.

---

# 🔍 Annotation Processing

A major component of the project was resolving issues with FoodSeg103's Supervisely bitmap annotations.

The notebook includes dedicated processing for:

* `meta.json` inspection
* Annotation inspection
* Image inspection
* Bitmap structure analysis
* Compressed bitmap decoding
* Segmentation mask generation
* Mask validation
* Diagnosis of the initial `IoU = 0.0000` issue

The corrected bitmap decoding pipeline enables the segmentation model to train using the actual pixel-level ingredient annotations.

---

# 🍳 Recipe Recommendation — Future Development

The next stage of the project will connect the detected ingredients to a recipe recommendation system.

The planned workflow is:

```text
Food Image
    │
    ▼
Ingredient Detection
    │
    ▼
Ingredient List
    │
    ├── Chicken
    ├── Tomato
    ├── Onion
    └── Garlic
    │
    ▼
Recipe Matching / AI Recommendation
    │
    ▼
Recommended Recipes
    │
    ├── Chicken Curry
    ├── Chicken Karahi
    └── Tomato Chicken
```

Potential implementation approaches include:

* Recipe dataset containing ingredient lists
* Ingredient-to-recipe matching
* Similarity-based recommendation
* Missing-ingredient analysis
* Recipe ranking based on detected ingredients
* LLM-based recipe generation
* Personalized recipe recommendations

This will transform the project from an ingredient segmentation system into an end-to-end **image-to-recipe AI application**.

---

# 🛠️ Tech Stack

## Machine Learning

* Python
* PyTorch
* Torchvision
* NumPy
* OpenCV
* Albumentations
* PIL
* Matplotlib

## Deep Learning

* Convolutional Neural Networks
* Encoder-Decoder Architecture
* Semantic Segmentation
* DeepLabV3+ experiments
* Batch Normalization
* Dropout
* Transposed Convolutions
* Gradient Clipping
* Mixed Precision Training

## Dataset

* FoodSeg103
* Supervisely annotations
* Bitmap segmentation masks

## Environment

* Google Colab
* NVIDIA GPU
* Tesla T4

---

# 🚀 Future Roadmap

### Phase 1 — Ingredient Detection

* [x] FoodSeg103 dataset integration
* [x] Supervisely bitmap decoding
* [x] Ingredient segmentation
* [x] PyTorch training pipeline
* [x] IoU evaluation
* [x] Model checkpointing
* [x] Training recovery

### Phase 2 — Recipe Recommendation

* [ ] Extract detected ingredient classes
* [ ] Build recipe dataset
* [ ] Create ingredient-recipe matching system
* [ ] Rank recipes based on available ingredients
* [ ] Identify missing ingredients
* [ ] Generate recipe suggestions
* [ ] Add AI/LLM recipe generation

### Phase 3 — End-to-End Application

* [ ] Image upload interface
* [ ] Ingredient visualization
* [ ] Recipe recommendation interface
* [ ] Recipe details and instructions
* [ ] Inference API
* [ ] Web application
* [ ] Deployment

---

# ⚠️ Current Status & Limitations

The current notebook is an experimental deep-learning training pipeline focused on **food ingredient detection and semantic segmentation**.

The best recorded validation IoU is:

```text
0.1446
```

The **recipe recommendation component is not yet implemented in the current notebook** and is planned as the second stage of the project.

The target values such as **0.3, 0.6, and higher** are experimental goals and should not be interpreted as achieved results.

---

# 👨‍💻 Author

**Muhammad Humair**

AI/ML & Automation Engineer

* Python
* Machine Learning
* Deep Learning
* Computer Vision
* Semantic Segmentation
* PyTorch
* FastAPI
* MLOps
* Automation

---

# ⭐ Project Highlights

```text
🍽️ AI Food Ingredient Detection
📊 FoodSeg103 Dataset
🧠 PyTorch Deep Learning
🎯 Pixel-Level Segmentation
🔍 Supervisely Bitmap Decoding
🚀 4,983 Training Images
🛡️ Crash-Resistant Training
📈 IoU-Based Evaluation
💾 Automatic Model Checkpointing
🍳 Ingredient-Based Recipe Recommendation
⚡ Google Colab GPU Optimized
```
