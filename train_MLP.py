"""
프로젝트: 펌프 부품 불량 검출 (Casting Defect Detection)
작성자 : 조영진 [Youngjin Jo]
작성일 : 2026/03/26
파일 역할: 준비된 데이터와 모델을 불러와서 실제로 AI를 학습시키는(Training) 코드
"""
# 1. 파이토치 도구 꺼내기
import torch
import torch.nn as nn
import torch.optim as optim # 최적화 도구(Optimizer) 모음집

# 그래픽카드를 사용할 수 있는지 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 현재 장착된 엔진: {device}")

# 2. 만든 파일에서 필요한 부분 가져오기
from dataset import train_loader        # 데이터 파일 ( 훈련용 )
from model_MLP import CastingMLP            # MLP 모델

# 3. 모델
model = CastingMLP()              # 모델이라는 변수에 만든 MLP모델 설정
model = model.to(device)          # 모델을 GPU 위로 보내기
# 4. 학습 기준 정하기
# AI가 낸 답과 실제 정답을 비교해서 채점하는 역할
criterion = nn.CrossEntropyLoss() # 분류문제에는 CrossEntropyLoss 가 적합
optimizer = optim.Adam(model.parameters(), lr=0.001) # 학습 속도 0.001 Adam이라는 최적화 도구 사용
# 5. 학습 시작
num_epochs = 5  # Epoch : 처음부터 끝까지 훑어보는 것. ( 5번 반복학습 )
print(f'학습을 시작합니다.')

for epoch in range(num_epochs):
    running_loss = 0.0  # Loss 기록을 위한 변수
    for i, (images, labels) in enumerate(train_loader): # i(배치 수)에 따른 image 사진 데이터 수 / labels (0 = 불량, 1 = 정상)
        images = images.to(device)  # 사진을 GPU로
        labels = labels.to(device)  # 정답지도 GPU로 
        # 1) 리셋 (기울기 초기화) - 이전 문제를 풀 때의 기억을 리셋
        optimizer.zero_grad()
        # 2) 문제 풀기 (순전파) - 사진 데이터를 모델에 넣어 예측값 (outputs) 얻기 
        outputs = model(images)
        # 3) 채점 (손실계산)- AI가 생각한 답 (outputs)와 실제 정답(labels)를 채점 (criterion) Loss 구하기
        loss = criterion(outputs, labels)
        # 4) 오답 분석 (역전파) - 왜 틀렸는지 거꾸로 타고 올라가며 원인 분석
        loss.backward()
        # 5) 수정 (파라미터 갱신) - 알아낸 원인을 바탕으로 오답을 줄이기 위해 가중치 조정
        optimizer.step()

        # 진행상황 출력 - 배치를 하나 풀 때마다 오답 점수를 출력
        running_loss += loss.item()

        # 배치를 10번 풀 때마다 중간점검
        if ( i + 1 ) % 10 == 0:
            print(f'[Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}] 오답 점수(Loss): {running_loss / 10:.4f}')
            running_loss = 0.0 # 출력 완료 했으니 점수 리셋

print('\n 학습 종료')

# 6. 가중치 를 파일로 영구 저장
torch.save(model.state_dict(), 'casting_mlp_brain.pth')
print(" 모델 가중치가 'casting_mlp_brain.pth' 파일로 저장되었습니다.")