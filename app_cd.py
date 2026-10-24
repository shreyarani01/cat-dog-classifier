import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload an image of a cat or a dog to test your trained PyTorch AlexNet model!")


class NotebookAlexNet(nn.Module):
    def __init__(self):
        super(NotebookAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2304, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=256, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model(model_path='cat_dog_cnn.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NotebookAlexNet()
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model, device

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Matches 2304 feature map size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Classify Image"):
        with st.spinner("Analyzing image..."):
            try:
                model, device = load_model('cat_dog_cnn.pth')
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    confidence, predicted_idx = torch.max(probabilities, 0)
                
                classes = ['Dog 🐶', 'Cat 🐱']
                predicted_label = classes[predicted_idx.item()]
                confidence_score = confidence.item() * 100
                
                st.success(f"**Prediction:** {predicted_label}")
                st.info(f"**Confidence:** {confidence_score:.2f}%")
                
            except Exception as e:
                st.error(f"Error loading model or running prediction: {e}")
