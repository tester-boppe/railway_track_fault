# railway_track_fault
This deep learning-based defect detection system classifies images as Defect or Non Defect using a fine-tuned EfficientNet-B0 model. Built with PyTorch and deployed via Streamlit, it allows real-time image classification. Optimized with AdamW, the model achieves 98% accuracy and is ready for deployment on Streamlit

# Defect Detection System

## Description
This deep learning-based defect detection system classifies images as **Defect** or **Non Defect** using a fine-tuned **EfficientNet-B0** model. Built with **PyTorch** and deployed via **Streamlit**, it allows real-time image classification. Optimized with **AdamW**, the model achieves **98% accuracy** and is ready for deployment on **Render**.

## Features
- Uses **EfficientNet-B0** for defect classification  
- Achieves **98% validation accuracy**  
- Implements **GradScaler** for mixed-precision training  
- Real-time predictions via **Streamlit** frontend  
- Deployable on **Render**  

## Technologies Used
- Python  
- PyTorch  
- Torchvision  
- Streamlit  
- Timm (pretrained models)  
- Streamlit (for deployment)  

## Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/your-repo.git
cd your-repo
pip install -r requirements.txt
