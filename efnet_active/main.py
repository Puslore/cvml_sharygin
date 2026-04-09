import torch
import cv2
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from collections import deque
import torch.nn as nn
from time import perf_counter
from pathlib import Path


save_path = Path.cwd()
MODEL_PATH_ALEX = save_path / 'alexnet.pth'
MODEL_PATH_EFNET = save_path / 'model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{DEVICE=}')


def build_AlexNet():
    weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    model = torchvision.models.alexnet(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False
    
    features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features=features, out_features=1)
    
    if MODEL_PATH_ALEX.exists():
        model.load_state_dict(torch.load(MODEL_PATH_ALEX, map_location=DEVICE))
    
    return model


def build_EfficientNet():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False
    
    features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=features, out_features=1)
    
    if MODEL_PATH_EFNET.exists():
        model.load_state_dict(torch.load(MODEL_PATH_EFNET, map_location=DEVICE))
    
    return model


alex_net = build_AlexNet()
ef_net = build_EfficientNet()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, .406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(frame):
    ef_net.eval()
    tensor = transform(cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    ))
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        predicted_ef = ef_net(tensor).squeeze()
        prob_ef = torch.sigmoid(predicted_ef).item()
        predicted_alex = alex_net(tensor).squeeze()
        prob_alex = torch.sigmoid(predicted_alex).item()
    
    label_ef = 'person' if prob_ef > 0.5 else 'no_person'
    label_alex = 'person' if prob_alex > 0.5 else 'no_person'
    
    return label_ef, prob_ef, label_alex, prob_alex


cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, 15)

while capture.isOpened():
    ret, frame = capture.read()
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == 'q':
        break
    
    label_ef, prob_ef, label_alex, prob_alex = predict(frame=frame)
    
    cv2.putText(frame, f"EFNet: {label_ef} ({prob_ef:.2f})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if label_ef == 'person' else (0, 0, 255), 2)
    cv2.putText(frame, f"AlexNet: {label_alex} ({prob_alex:.2f})", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if label_alex == 'person' else (0, 0, 255), 2)
        
    cv2.imshow('Camera', frame)


capture.release()
cv2.destroyAllWindows()
