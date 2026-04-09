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
# print(alex_net)
ef_net = build_EfficientNet()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    params=filter(lambda p: p.requires_grad,
                            ef_net.parameters()),
    lr=0.0001
)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, .406],
        std=[0.229, 0.224, 0.225]
    )
])


def train(buffer):
    if len(buffer) < 10:
        return None
    ef_net.train()
    images, labels = buffer.get_batch()
    optimizer.zero_grad()
    predictions = ef_net(images).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


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


class Buffer():
    def __init__(self, max_size=16) -> None:
        self.frames = deque(maxlen=max_size)
        self.labels = deque(maxlen=max_size)
    
    def append(self, tensor, label) -> None:
        self.frames.append(tensor)
        self.labels.append(label)
    
    def __len__(self):
        return len(self.frames)
    
    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels), dtype=torch.float32)
        
        return images, labels


cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, 15)

buffer = Buffer()
count_labeled = 0


while capture.isOpened():
    ret, frame = capture.read()
    key = chr(cv2.waitKey(1) & 0xFF)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if key == 'q':
        break
    
    elif key == '1': # person
        tensor = transform(image)
        buffer.append(tensor=tensor, label=1.0)
        count_labeled+=1
    elif key == '2': # no person
        tensor = transform(image)
        buffer.append(tensor=tensor, label=0.0)
        count_labeled+=1
    elif key == 'p': # perdict
        t = perf_counter()
        label_ef, prob_ef, label_alex, prob_alex = predict(frame=frame)
        print(f'\nElapsed time: {perf_counter() - t}')
        print(f'EFNet  : {label_ef} ({prob_ef})')
        print(f'AlexNet: {label_alex} ({prob_alex})')
    elif key == 's': # save model
        torch.save(ef_net.state_dict(), MODEL_PATH_EFNET)
    
    print(f'{len(buffer)=}')
    if count_labeled >= buffer.frames.maxlen:
        loss = train(buffer=buffer)
        if loss:
            print(f'Loss = {loss}')
        count_labeled = 0

    cv2.imshow('Camera', frame)


capture.release()
cv2.destroyAllWindows()
