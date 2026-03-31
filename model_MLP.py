"""
프로젝트 : 펌프 부품 불량 검출 (Casting Defect Detection)
작성자 : 조영진 [Youngjin Jo]
작성일 : 2026/03/26
파일 내용  : 사진을 보고 불량인지 아닌지 판단하는 MLP(Multi-Layer Perceptron) 구조 설계도
            학습이 목표이기에 최적화 모델이 아닌 여러가지 모델을 시도해봄으로써 코드를 짜보는 경험과
            데이터셋에 어울리는 모델을 직접 확인해보기 위한 코드.
"""
'''
MLP(Multi-Layer Perceptron) 다층 퍼셉트론 
3가지 층(Layer) 구조 : 데이터를 처음 받아들이는 입력층 (Input Layer) / 중간에서 계산해서 특징을 뽑아내는 은닉층 (Hidden Layer) / 최종 판단(정상/불량)하는 출력층(OutPut Layer)
Fully Connected ( 모두 연결된 구조 ) : 앞 층의 모든 노드(점)가 다음 층의 모든 노드와 빠짐없이 촘촘하게 연결되어 있음. (nn.Linear의 역할)
활성화 함수 : 은닉층을 통과할 때마다 ReLU 같은 활성화 함수를 씌워 비선형성 학습을 가능하게 함.
 ## 무조건 1차원 형태 ( view 또는 nn.Flatten() 를 통해 중간과정 없이 다른 차원의 데이터도 1차원 변경 가능 ) 
'''
# 1. 파이토치 도구 꺼내기
import torch.nn as nn               # Neural Network 파이토치의 신경망 가져오기
import torch.nn.functional as F     # 활성화 함수 등 함수들이 모여있는 꾸러미

# 2. 다층 퍼셉트론 (MLP) 만들기
class CastingMLP(nn.Module):            # nn.Module 상속받기
    def __init__(self):
        super(CastingMLP, self).__init__()
        self.fc1 = nn.Linear(300*300, 512) # dataset에서 transforms.Resize 를 했던 300x300 크기를 받아서 원하는 갯수인 512개의 특징으로 뽑아낸다.
        self.fc2 = nn.Linear(512,256)      # 512개의 특징을 256개로 더 추려내기
        self.fc3 = nn.Linear(256,2)        # 256개의 특징을 최종적으로 2개 ( 정상 확률, 불량 확률 ) 뽑아냄

# 3. 데이터 작동 순서 정하기
    def forward(self, x):           # x 는 dataloader로 넘겨줄 이미지 데이터
        x = x.view(-1, 300*300)     # view(-1) = batch 개수는 알아서 계산하게 끔 유도 / 300*300은 MLP가 계산할 수 있게 펴질 길이
        x = F.relu(self.fc1(x))     # 첫번째 fc1을 통과 시키고 ReLU 0보다 작은 건 버리고 0보다 큰 값만 살리는 활성화 함수
        x = F.relu(self.fc2(x))     # 두번째 fc2를 통과 시키고 ReLU
        x = self.fc3(x)             # 마지막 fc3을 통과시켜 최종 값 2개를 만들어 냄
        return x                    # 최종 계산된 결과 ( 정상 / 불량 판단 숫자 2개 )를 밖으로 꺼냄

print(f'MLP 설계도 완성')