"""
프로젝트 : 펌프 부품 불량 검출 (Casting Defect Detection)
작성자 : 조영진 [Youngjin Jo]
작성일 : 2026/03/29
파일 내용  : 사진을 보고 불량인지 아닌지 판단하는 CNN 구조 설계도
            지금 분석하는 데이터셋은 MLP와는 전혀 어울리지 않는 데이터라는 것을 확인.
            공간의 특징을 파악할 수 있는 CNN 모델로 다시 분석
"""
'''
CNN (Convolutional Neural Network) 합성곱 신경망
3가지 핵심 층(Layer) 구조:
합성곱층 (Convolution Layer): 이미지에서 특징(선, 윤곽, 질감 등)을 도장(필터) 찍듯 훑으며 찾아내는 핵심 층.
풀링층 (Pooling Layer): 데이터의 크기를 반토막 내듯 줄이면서, 가장 강하고 중요한 특징만 남겨서 요약하는 층. (주로 가장 큰 값을 뽑는 Max Pooling 사용)
완전연결층 (Fully Connected Layer): 앞에서 다듬어온 특징들을 모아서 최종적으로 이게 무엇인지(정상/불량, 개/고양이 등) 판단하는 마무리 층.

부분 연결과 가중치 공유 (Local Connectivity & Weight Sharing):
작은 돋보기(필터/커널)로 이미지를 부분부분 겹치게 스캔하는 방식 / 같은 필터를 계속 재사용(가중치 공유) 파라미터 수가 확 줄어듬. (nn.Conv2d의 역할)

활성화 함수: 합성곱층에서 특징을 뽑아낸 뒤에 ReLU 같은 활성화 함수를 씌워서 비선형성을 학습.

## 데이터 형태 유지 (2차원/3차원):
처음부터 무작정 1차원으로 펴버리는 MLP와 달리, 이미지의 가로, 세로, 색상(채널) 형태를 그대로 유지하면서 공간적인 정보를 학습.
단, 마지막에 최종 판단을 내리기 직전(완전연결층에 넣을 때)에만 view나 nn.Flatten()을 써서 1차원으로 펴줌.
'''


import torch.nn as nn
import torch.nn.functional as F

class CastingCNN(nn.Module):
    def __init__(self):
        super(CastingCNN, self).__init__()
        # 첫번째 돋보기
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # in_channels = 1 흑백사진 이기때문에 1 // CNN은 컬러도 가능하지만 불량을 찾을 땐 형태와 명암을 찾고 , 데이터도 작게해서 속도도 빠르게 하기 위해서 흑백으로 사용.
        # out_channels = 16 돋보기 종류를 16개로 늘려서 특징을 다양하게 찾음
        # kernel_size = 3 // 3x3 픽셀의 돋보기를 사용
        # padding = 1 모서리 부분 데이터가 날아가지 않게 여백

        # 첫번째 요약본 (Pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2x2 크기로 사진을 묶어서 가장 강한 특징 (Max) 만 크기를 절반으로 줄임
        # 300x300 사이즈의 사진이 여기서 150x150 으로 작아짐.

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 위에서의 in_channel은 흑백의 채널을 받았기 때문에 1이지만 두번째의 in_channel에서는 처음 self.conv1에서의 out_channel (구별하기 위한 특징)을 가져옴.
        # out_channel은 16에서 더 늘려서 32개 / 더 세밀한 특징을 찾아냄

        self.fc1 = nn.Linear(32 * 75 * 75,64)
        self.fc2 = nn.Linear(64,2)
        # 두번의 Pooling 을 거치면 사진 크기가 300x300 -> 150x150 => 75x75가 되기 때문에 out_channel * size * size 를 해줌
        # 뒷쪽의 64는 직접 정하는 압축을 원하는 숫자 // 보통 2의 제곱수를 사용하지만 너무 적거나 많으면 안되기 때문에 64나 128부터 시작하는 게 일반적

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # 첫번째 conv1을 보고 활성화 함수 relu 로 특징 강조 후 요약
        x = self.pool(F.relu(self.conv2(x)))    # 두번째 conv2를 보고 반복
        x = x.view(-1, 32*75*75)                # 2차원 데이터로는 결론을 내릴 수 없기 때문에 1줄로 변환
        x = F.relu(self.fc1(x))                 # fc1, fc2를 거쳐 최종 정답 뱉어내기 
        x = self.fc2(x)

        return x
    
if __name__ == "__main__":
    model = CastingCNN()
    print("CNN 2차원 모델 설계도 완성")
    print(model) 