"""
프로젝트: 펌프 부품 불량 검출 (Casting Defect Detection)
작성자 : 조영진 [Youngjin Jo]
작성일 : 2026/03/28
파일 역할: 훈련이 끝난 AI를 가져와서, 처음 보는 사진(Test 데이터)으로 실력 테스트하기
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CastingMLP
from sklearn.metrics import confusion_matrix, classification_report # 심화 데이터를 추출하기 위한 사이킷런 추가
# 1. GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 테스트 데이터 준비 ( 훈련과 똑같은 규격 )
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((300,300)),
    transforms.ToTensor()
])

# 3. test 폴더에서 테스트 파일 가져오기
test_dataset = datasets.ImageFolder(r'casting_data\casting_data\test', transform = transform)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)    # 테스트는 섞지 않아도 됨

# 4. 학습한 AI 불러오기
model = CastingMLP()
model = model.to(device)

# train 에서 학습한 가중치 파일 (.pth) 를 빈 곳에 덮어주기
model.load_state_dict(torch.load('casting_mlp_brain.pth'))

# 평가모드
model.eval()

# 5. 채점
correct = 0     # 맞춘 문제 수
total = 0       # 전체 문제 수

all_labels = []     # AI가 제출한 답안지와 실제 정답지를 모아둘 바구니
all_preds = []

print(f'채점을 시작합니다.\n')

# 채점할 땐 기울기 계산을 하면 안되기 때문에 기능을 끔.
with torch.no_grad():
    for images, labels in test_loader:      # 사진과 정답지 가져오기
        images = images.to(device)          # 사진과 정답지도 GPU로 보내기
        labels = labels.to(device)

        outputs = model(images)             # 사진을 보여주고 예측값(outputs) 받아내기
        predicted = torch.max(outputs.data, 1)[1] # torch.max (최대값, 인덱스) 를 할당받는데 인덱스만 사용하기때문에 [1]로 인덱스만 가져오기

        total += labels.size(0) # 전체 문제 수 누적
        correct += (predicted == labels).sum().item()

        # 계산이 끝난 정답과 예측값을 CPU로 다시 내린 다음에 바구니에 담기
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
# 6. 최종결과 출력
accuracy = 100 * correct / total
print(f'데이터 정확도(Accuracy): {accuracy:.2f}%')

#=====================================================================================================#
# 사이킷런 도구에 넣고 심화데이터 추출
print("=" * 20)
print("[심화 데이터: 오차 행렬 (Confusion Matrix)]")
# 행렬에 들어갈 클래스 이름 (0: 불량, 1: 정상)
print(confusion_matrix(all_labels, all_preds))
print("=" * 20)

print("📈 [심화 성적표: 상세 리포트]")
# 각 항목별 정밀도, 재현율, F1-스코어를 깔끔하게 표로 출력
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("=" * 20)
print("F1-score 기준 공식:")
print(f"   $2 \\times TP / (2 \\times TP + FP + FN)$")
print("=" * 20)