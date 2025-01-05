import io

from PIL import Image
import torch
from pathlib import Path

from config.config import config
from src.model import VisionTransformer


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="vit.pth")
    """

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def load_model(model_path: str):
    model = VisionTransformer(num_classes=5)
    transforms = model.get_transforms()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, transforms


def predict(model, image_bytes: bytes, preprocess):
    """
    Makes predictions using the provided model and image data.

    Args:
        model: The Vision Transformer model.
        image_bytes: The image data in bytes format.
        preprocess: The transformation function for preprocessing the image.

    Returns:
        A dictionary with predicted label and confidence.
    """
    # Load and preprocess the image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        label = config["label_classes"][predicted_class]

        confidence = probabilities[predicted_class].item()

    return {"label": str(label), "probability": confidence}
