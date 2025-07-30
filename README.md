# CAPTCHA Character Recognition using Random Forest

This project addresses the problem of recognizing characters from simple CAPTCHA images using classical machine learning methods. A Random Forest classifier was trained on manually segmented and preprocessed CAPTCHA characters to perform character-level recognition with high accuracy.

## 🧩 Problem Framing

The objective is to accurately identify all characters in fixed-format CAPTCHA images. The main challenges include:

- Segmenting characters from noisy backgrounds with horizontal gray artifacts.
- Handling class imbalance across many character classes.
- Designing a pipeline that requires minimal preprocessing yet performs robustly on high-dimensional flattened pixel data.

## 🛠️ Solution Overview

**Data Preprocessing & Segmentation**  
- Characters were segmented using fixed x-coordinate slices based on consistent spacing observed in images (e.g., segments like (5,13), (14,22), etc.).  
- Horizontal gray lines were removed by thresholding pixel channel differences and intensities (e.g., abs(R-G)<50, intensity between 50-260), replacing them with white pixels.  
- Pixels always white across the dataset were masked out to reduce input dimensionality and noise. This mask was saved and reused during inference.  
- Each segmented character image was flattened into a feature vector for classification.

**Model Training & Selection**  
- A Random Forest classifier was chosen for its robustness to noise, ability to handle class imbalance via ensemble voting, and minimal need for feature scaling or extensive tuning.  
- Hyperparameters tuned included number of trees (`n_estimators`=100), tree depth (`max_depth`=None), and minimum samples split/leaf.  
- The model was trained on 70% of the dataset, with 30% held out for evaluation.

**Model Evaluation**  
- The final model achieved approximately 94% accuracy on the test set and a weighted F1 score of about 0.93.  
- The confusion matrix showed excellent performance across most characters, with minor confusion between visually similar ones (e.g., O vs 0, I vs 1).  

**Model Persistence & Inference**  
- The trained Random Forest model was saved as `random_forest_model.pkl` for reproducibility and deployment.  
- The pixel mask (`always_white_mask.pkl`) was also saved to ensure consistent feature filtering during inference.  
- The inference pipeline loads an unseen CAPTCHA image, segments it into characters, cleans noise, applies the saved mask, and predicts each character using the trained model. Predictions are saved to a text file.

## 📦 File Structure

- ├── IMDA_model_training.ipynb   # Training and tuning notebook
- ├── random_forest_model.pkl     # Saved trained model
- ├── always_white_mask.pkl       # Mask of always-white pixels
- ├── input100.jpg                # Example unseen CAPTCHA image
- ├── predicted100.txt            # Prediction output from example image
- ├── captcha_characters_dataset.csv # Flattened feature dataset CSV
- ├── IMDA_captcha_inference.py   # Inference pipeline class
- ├── README.md                   # Project documentation

## 💡 Recommendations & Next Steps

- **Expand Dataset:** Collect more labeled CAPTCHA images to improve model generalization, especially for underrepresented characters.  
- **Balance Classes:** Apply oversampling techniques like SMOTE or class weighting to address class imbalance more effectively.  
- **Explore Deep Learning:** Implement CNNs or hybrid models to handle more complex CAPTCHA styles and distortions.  
- **Dynamic Segmentation:** Move from fixed coordinate cropping to object detection or contour-based segmentation for flexible and robust character extraction.

## 🔧 Usage

### Files Required
Make sure the following files are in the same directory:
- `captcha_inference.py` – main script for inference
- `random_forest_model.pkl` – trained Random Forest model
- `always_white_mask.pkl` – boolean mask to filter always-white pixels
- `input100.jpg` – sample image for prediction (or replace with your own)

### Run the script

To run the CAPTCHA prediction:

```bash
python captcha_inference.py
