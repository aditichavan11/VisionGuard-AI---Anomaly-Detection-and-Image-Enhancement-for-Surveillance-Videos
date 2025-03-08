import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load SRGAN model
try:
    srgan_model = load_model("models/srgan_generator.h5")
    print("✅ SRGAN Model Loaded Successfully")
except Exception as e:
    print("❌ Error Loading Model:", str(e))

def enhance_image(image_path):
    """Enhances an image using the SRGAN model."""
    # Verify file existence
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None  # Avoid crashing if file is missing

    try:
        print(f"🔹 Processing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Error: Image could not be read (Check format or corrupted file)")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))  # Resize to match SRGAN input size
        image = image.astype(np.float32) / 255.0  # Normalize

        # Expand dimensions to match model input
        image = np.expand_dims(image, axis=0)

        # Predict enhanced image
        enhanced_image = srgan_model.predict(image)[0]  # Remove batch dimension

        # Convert back to uint8 format
        enhanced_image = np.clip(enhanced_image * 255.0, 0, 255).astype(np.uint8)

        # Save the enhanced image
        enhanced_image_path = image_path.replace(".jpg", "_enhanced.jpg") if ".jpg" in image_path else image_path.replace(".png", "_enhanced.png")
        cv2.imwrite(enhanced_image_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
        print(f"✅ Enhanced image saved at: {enhanced_image_path}")

        return enhanced_image_path

    except Exception as e:
        print(f"⚠ Enhancement failed: {str(e)}")
        return None
