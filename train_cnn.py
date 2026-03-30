import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model_CNN import CastingCNN

# 1. 데이터 준비 ( 이미지 변환 )
transform = transforms.Compose([
    transforms.Grayscale(),         # 흑백
    transforms.Resize((300,300)),   # 크기는 설계도 만들었을 때 300x300으로 가정해두었기 때문에 300x300
    transforms.ToTensor(),
])

# 데이터셋 경로
train_dataset = datasets.ImageFolder(root=r'casting_data\casting_data\train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. 모델, 손실함수, 최적화 도구 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CastingCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# 3. 학습 시작
print(f'CNN 모델 학습 시작')

for epoch in range(5):              # 5회 반복
    model.train()                   # 학습모드 / eval() 은 평가모드
    running_loss = 0.0              # 오답갯수 초기화
    for images, labels in train_loader:             # 아까 설정해둔 32개 묶음으로 사진과 정답을 가져옴
        images, labels = images.to(device), labels.to(device)       # 빠른 학습을 위해 GPU로 이동

        optimizer.zero_grad()               # 기울기 초기화
        outputs = model(images)             # 사진을 보고 예측값 outputs 가져오기
        loss = criterion(outputs, labels)   # 예측값(outputs)과 정답(labels)를 비교해서 Loss 측정
        loss.backward()                     # 역전파 - 뒤에서부터 추적해가며 왜 틀렸는지 확인
        optimizer.step()                    # 오답분석결과를 바탕으로 앞으로 틀리지 않게 업데이트

        running_loss += loss.item()         # 오답 점수 누적

    print(f'Epoch [{epoch+1}/5], Loss : {running_loss/len(train_loader):.4f}') 


# 4. 저장하기
torch.save(model.state_dict(), 'casting_CNN_model.pth')
print('학습 완료 및 저장')