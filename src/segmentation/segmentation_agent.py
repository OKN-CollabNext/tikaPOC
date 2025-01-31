#!/usr/bin/env python
import os
import torch
import numpy as np
from PIL import Image
from segmentation_main.segmentation_demo import SemanticSegmentation, DLStudio

class SegmentationAgent:
    def __init__(self):
        # Initialize DLStudio configuration with the given parameters.
        self.config = DLStudio(
            dataroot="./data/",
            image_size=[64, 64],
            path_saved_model="./saved_model",
            momentum=0.9,
            learning_rate=1e-4,
            epochs=6,
            batch_size=4,
            classes=('cake', 'dog', 'motorcycle'),
            use_gpu=torch.cuda.is_available()
        )

        # Create the segmentation instance (with a maximum of 5 objects per image).
        self.segmenter = SemanticSegmentation(dl_studio=self.config, max_num_objects=5)

        # Build the U-Net–like model (mUnet) with skip connections enabled.
        self.model = self.segmenter.mUnet(skip_connections=True, depth=12)

        # Load the pretrained model weights (if available).
        self._load_pretrained_model()

        # Set the model to evaluation mode.
        self.model.eval()

    def _load_pretrained_model(self):
        model_file = self.config.path_saved_model
        if os.path.exists(model_file):
            try:
                # Load the state dictionary on the CPU.
                state_dict = torch.load(model_file, map_location=torch.device("cpu"))
                self.model.load_state_dict(state_dict)
                print("Model loaded successfully!")
            except Exception as err:
                print(f"Could not load pretrained model. Error: {err}")
        else:
            print(f"Pretrained model file not found at {model_file}.")

    def segment_image(self, input_image: Image.Image) -> Image.Image:
        """
        Accepts a PIL image, processes it, and returns a grayscale segmentation mask.
        """
        # If the image is in RGBA mode, convert it to RGB.
        if input_image.mode == 'RGBA':
            input_image = input_image.convert('RGB')

        # Resize the image to the expected input size.
        target_size = (64, 64)
        resized_img = input_image.resize(target_size)

        # Convert the resized image to a numpy array and then rearrange axes to (C, H, W).
        img_np = np.array(resized_img)
        img_np = img_np.transpose(2, 0, 1)

        # Create a float tensor from the numpy array and add a batch dimension.
        img_tensor = torch.tensor(img_np, dtype=torch.float).unsqueeze(0)

        # Run inference without gradient tracking.
        with torch.no_grad():
            pred_tensor = self.model(img_tensor)
            pred_tensor = pred_tensor.squeeze(0)  # Remove the batch dimension

        # For multi-channel output, use argmax along the channel axis.
        if pred_tensor.shape[0] > 1:
            pred_np = torch.argmax(pred_tensor, dim=0).cpu().numpy()
        else:
            # If the output is single channel, simply remove the channel dimension.
            pred_np = pred_tensor.squeeze(0).cpu().numpy()

        # Normalize the prediction to the 0-255 range.
        max_val = pred_np.max() if pred_np.max() > 0 else 1
        norm_pred = (pred_np * 255.0 / max_val).astype('uint8')

        # Convert the normalized numpy array into a grayscale PIL Image.
        mask_image = Image.fromarray(norm_pred, mode='L')
        return mask_image

# For module-level testing (if desired)
if __name__ == "__main__":
    # Example test run:
    agent = SegmentationAgent()
    # Open an example image (replace 'your_image.jpg' with an actual file path)
    test_image = Image.open("your_image.jpg")
    mask = agent.segment_image(test_image)
    mask.show()
