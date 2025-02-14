import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from huggingface_hub import hf_hub_download

# Hugging Face model details
REPO_ID = "lasya-boppe/track_fault"  # Update with your correct repo ID
MODEL_FILENAME = "model_98_accuracy.pth"


# Load the trained model
class DefectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.backbone(x)


# Load model weights from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, local_dir="./")

    model = DefectModel()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])  # Ensure correct key
    model.eval()
    return model


model = load_model()


# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# Streamlit UI
st.title("Defect Detection Model")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    class_names = ["Defect", "Non Defect"]
    st.write(f"Prediction: **{class_names[predicted.item()]}**")
