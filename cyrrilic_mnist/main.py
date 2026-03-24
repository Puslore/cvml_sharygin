import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Приношу извинения, если полетит импорт, но на винде работает только так(
from train_model import CyrillicCNN, test_loader, dataset

# Вот в теории рабочий вариант
# from cyrrilic_mnist.train_model import CyrillicCNN, test_loader, dataset


cur_path = Path(__file__).parent
model_path = cur_path / 'cyrillic_model.pth'
if not model_path.exists():
    raise RuntimeError(f'Model not trained! {model_path}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device=}')

model = CyrillicCNN().to(device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

cyrillic_path = cur_path / 'cyrillic' / 'Cyrillic'

correct = 0
total = 0


with torch.no_grad():
    for i in range(1024):
        it = iter(test_loader)
        images, labels = next(it)
        image = images[i].unsqueeze(0)
        image = image.to(device)

        output = model(image)
        
        _, predicted = torch.max(output, 1)

        if dataset.alphabet[labels[i]] == dataset.alphabet[int(predicted.item())]:
            correct += 1
        total += 1
        if i > 1000:
            print(f'True - {dataset.alphabet[labels[i]]} ::: Pred - {dataset.alphabet[int(predicted.item())]}')
print('Accuracy: ', correct/total * 100)
