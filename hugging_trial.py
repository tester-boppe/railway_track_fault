import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from huggingface_hub import hf_hub_download
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Hugging Face model details
REPO_ID = "lasya-boppe/track_fault"  # Update with your correct repo ID
MODEL_FILENAME = "model_98_accuracy.pth"

# Email credentials
SENDER_EMAIL = "tester.boppe.101@gmail.com"
PASSWORD = "toqb suaa xqpi dsyx"

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

# Function to send email (without image attachment)
def send_email(receiver_email, detection_time):
    user_name = receiver_email.split("@")[0].capitalize()  # Extract username and capitalize

    subject = "Defect Alert 🚨"
    body = f"""
    Hello {user_name},

    A defect has been detected in the uploaded image at **{detection_time}**.

    Please take necessary action immediately.

    Best regards,  
    **Defect Detection System**  
    """

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, PASSWORD)
            server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

# Streamlit UI
st.title("Defect Detection Model")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    class_names = ["Defect", "Non Defect"]
    prediction = class_names[predicted.item()]
    st.write(f"Prediction: **{prediction}**")

    # If a defect is detected, ask for an email
    if prediction == "Defect":
        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Capture the current timestamp
        st.warning(f"A defect has been detected at **{detection_time}**! Please enter your email to receive an alert.")

        email = st.text_input("Enter your email address")

        if email:
            if st.button("Send Alert"):
                send_email(email, detection_time)
