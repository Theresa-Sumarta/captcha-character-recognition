import numpy as np
import joblib
from PIL import Image

class Captcha(object):
    def __init__(self):
        # Load the saved Random Forest model
        self.model = joblib.load("random_forest_model.pkl")

        # Load the always-white mask used during training
        self.always_white_mask = joblib.load("always_white_mask.pkl")

        # Fixed character segmentation coordinates
        self.x_coords = [(5, 13), (14, 22), (23, 31), (32, 40), (41, 49)]

    def load_jpg_image(self, img_path):
        """Load .jpg image and convert it into NumPy array (H, W, 3)."""
        img = Image.open(img_path).convert('RGB')
        return np.array(img)

    def manual_segment(self, img):
        """Segment the image using fixed x-coordinates."""
        return [img[:, x1:x2] for (x1, x2) in self.x_coords]

    def clean_greylines(self, segment, gray_thresh=50, intensity_range=(50, 260)):
        """Remove grayish horizontal artifacts."""
        seg = segment.copy()
        gray = np.abs(seg[:, :, 0] - seg[:, :, 1]) < gray_thresh
        gray &= np.abs(seg[:, :, 1] - seg[:, :, 2]) < gray_thresh
        intensity = seg.mean(axis=2)
        mask = gray & (intensity >= intensity_range[0]) & (intensity <= intensity_range[1])
        seg[mask] = [255, 255, 255]
        return seg

    def flatten_and_filter(self, seg_img):
        """Flatten image and remove always-white columns."""
        flattened = seg_img.flatten()
        filtered = flattened[~self.always_white_mask]
        return filtered

    def __call__(self, im_path, save_path):
        """Perform inference on unseen CAPTCHA image."""
        # Load image
        image = self.load_jpg_image(im_path)

        # Segment into 5 characters
        segments = self.manual_segment(image)

        # Clean gray lines and predict each character
        predicted_text = ""
        for seg in segments:
            cleaned = self.clean_greylines(seg)
            features = self.flatten_and_filter(cleaned)
            prediction = self.model.predict([features])[0]
            predicted_text += prediction

        # Save the predicted string
        with open(save_path, "w") as f:
            f.write(predicted_text)

if __name__ == "__main__":
    captcha_char = Captcha()
    captcha_char("input100.jpg", "predicted100.txt")
