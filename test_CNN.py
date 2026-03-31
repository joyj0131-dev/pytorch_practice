"""
프로젝트 : 펌프 부품 불량 검출 (Casting Defect Detection)
작성자 : 조영진 [Youngjin Jo]
작성일 : 2026/03/31
파일 내용  : 학습한 CNN 모델을 불러와서 최종 성능을 평가하는 코드
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model_CNN import CastingCNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# 1. 테스터용 데이터 준비 ( 학습과 똑같은 규격 )
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((300,300)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(root=r'casting_data\casting_data\test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. 학습한 내용 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CastingCNN().to(device)
model.load_state_dict(torch.load('casting_CNN_model.pth')) # train 완료 한 파일
model.eval() # 평가모드 .eval()

print('CNN 모델 테스트 및 오차행렬 분석')

# 3. 채점 시작
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad(): # 기울기 계산 금지
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # 2개 점수 중 더 높은 걸 정답으로 가져옴
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy()) # 오차행렬을 위해 AI가 예상한 답과 실제 정답들을 모아두기
        all_labels.extend(labels.cpu().numpy())

# 4. 결과
acuuracy = 100 * correct / total
print(f'\n 최종 CNN 테스트 정확도 : {acuuracy:.2f}%')

# 5. 오차행렬 분석 및 저장
cm = confusion_matrix(all_labels,all_preds)
print(f'\n 오차행렬 (Confusion Matrix) :')
print(cm)

os.makedirs('./result', exist_ok=True) # result 폴더가 없으면 오류가 나기에 미리 만들기

# 사진으로 저장 ( 보고서 .md 작성용 )
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('AI predicted')
plt.ylabel('Actual Truth')
plt.title(f'CNN Confusion Matrix (Acc: {acuuracy:.2f}%)')
plt.savefig(r'./result/cnn_confusion_matrix.png')
print('오차행렬 이미지가 result 폴더에 저장 완료 되었습니다.')