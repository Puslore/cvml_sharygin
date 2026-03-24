import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split


SAVE_PATH = Path(__file__).parent

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f'{device=}')

torch.manual_seed(42)


class CyrillicDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        num_classes = -1
        self.dataset = []
        self.alphabet: str = ''
        # загрузка всех изображений букв из подпапок с вычислением количества классов
        for cls in sorted(path.glob('*')):
            if str(cls.name)[0] != '.':
                num_classes += 1
                # сохранение букв алфавита для последующего использования
                self.alphabet += str(cls)[-1]
                # загрузка путей к PNG файлам и привязка к номеру класса
                for letter in cls.glob('*.png'):
                    self.dataset.append((str(letter), num_classes))

    def __getitem__(self, index):
        # открытие изображения при запросе
        image = Image.open(self.dataset[index][0])
        # ресайз, конвертация в тензор, аугментация
        augments = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.RandomAffine(5, (0.1, 0.1), (0.5, 1), 10)
        ])
        return augments(image), self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)


class CyrillicCNN(nn.Module):
    def __init__(self):
        super(CyrillicCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=256 * 2 * 2, out_features=34)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        
        return x


cyrillic_path = SAVE_PATH / 'cyrillic' / 'Cyrillic'
dataset = CyrillicDataset(path=cyrillic_path)

train_dataset, test_dataset = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42
)

_batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=_batch_size,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=_batch_size * 16,
    shuffle=False
)
print(f'{len(train_loader)=}, {len(test_loader)=}')


if __name__ == "__main__":
    model = CyrillicCNN().to(device=device)
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    print(f'{total_params=}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), 
                                        lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=10,
        gamma=0.1
    )


    num_epochs = 30
    train_loss: list[float] = []
    train_acc: list[float] = []
    test_loss: list[float] = []
    test_acc: list[float] = []

    model_path = SAVE_PATH / 'cyrillic_model.pth'

    for epoch in range(num_epochs):
        model.train()
        run_loss: float = 0.0
        total: int = 0
        correct: int = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device=device), labels.to(device=device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        epoch_loss = run_loss / len(train_loader)
        epoch_acc = 100 * (correct / total)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
        # Оценка на тестовом наборе
        model.eval()
        test_run_loss: float = 0.0
        test_total: int = 0
        test_correct: int = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device=device), labels.to(device=device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_run_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        epoch_test_loss = test_run_loss / len(test_loader)
        epoch_test_acc = 100 * (test_correct / test_total)
        test_loss.append(epoch_test_loss)
        test_acc.append(epoch_test_acc)
        
        print(f'{epoch=}, {epoch_loss=:.3f}, {epoch_acc=:.3f}')

    torch.save(model.state_dict(), model_path)

    plt.figure()
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.tight_layout()
    plt.savefig(SAVE_PATH / 'train.png', dpi=150, bbox_inches='tight')
    plt.show()
