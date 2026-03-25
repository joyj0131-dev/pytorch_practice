"""
프로젝트 : 펌프 부품 불량 검출 (Casting Defect Detection)
작성자 : 조영진 [Youngjin Jo]
작성일 : 2026/03/26
파일 내용  : 캐글 이미지 데이터셋을 불러오고, AI 모델이 학습할 수 있도록 
            텐서(Tensor)로 변환 및 전처리(Resize, Grayscale)하는 코드입니다.
"""


# 1. 도구 꺼내기
import os
from torchvision import datasets, transforms # 직접 다운 받은 파일도 datasets 이 필요.
                                             # transforms은 사진을 다루기 쉽게 전처리 하는 도구
from torch.utils.data import DataLoader # 데이터를 모델에 조금씩 잘라서 주는 역할

# 2. 데이터 폴더 경로 설정
data_dir = './casting_data/casting_data' # .은 현재 파이썬 파일 / 데이터가 있는 폴더의 상대경로를 찾아 작성하기

# 3. 이미지 변환 ( 전처리 레시피 )
# Compose는 여러 변환 단계를 한개로 묶어준다.
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                                # 불량은 색깔보다 질감이 중요하기 때문에 RGB값을 없애서 계산을 빠르게 함
                                transforms.Resize((300,300)),   # AI가 헷갈리지 않게 사진의 크기를 가로세로 300픽셀 동일하게 변경
                                transforms.ToTensor()])     # 파이토치가 계산할 수 있는 Tensor로 변경

# 4. Dataset 만들기
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
# train 폴더의 def , ok 정상과 불량 이라는 라벨을 스스로 읽고 transform 과정에서 준비했던 내용을 전부 적용시킴

# 5. DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
# 한번에 32장씩 묶어서 (batch) 넘길 수 있게 만듬
# shuffle : 정상과 불량 사진이 편향되지 않도록 순서를 섞음 ( 순서를 보고 답을 외우는 것도 방지 )

# 확인과정

print(f'준비된 학습용 이미지 개수 : {len(train_dataset)}장')
print(f'분류할 클래스 (정상 / 불량) : {train_dataset.classes}') # def_front 와 ok_front 로 폴더 이름을 보고 정답지 목록을 작성함
                                                              # 알파벳 순으로 d , o 정리
                                                              # 인덱스 0 def (불량) / 인덱스 1 ok (정상) 으로 나누어주면 그것을 보고 판단