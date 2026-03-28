프로젝트 : 펌프 부품 불량 검출 (Casting Defect Detection)
작성자 : 조영진 [Youngjin Jo]
작성일 : 2026/03/28
파일 내용  : 불량 검출을 위한 AI모델 만들기 중 MLP 모델에 대한 후기.

AI딥러닝 공부를 하기위해 프로젝트를 시작했고, 처음이라 막막하기때문에 LLM을 사용하여 기초부터 어떻게 흘러가는 지 흐름을 배우는 시간이었습니다.
'펌프 부품 불량 검출' 같은 프로젝트에선 MLP같은 모델보단 CNN같은 모델이 더 효율이 좋은 것은 알고 있지만
공부가 목적이기때문에 MLP를 사용하는 코드를 작성해보았습니다.
# Epoch 확인
![Epoch check](<./result/CDD(MLP_ver1 epoch=5 lr=0.001).png>)
![Epoch check](<./result/CDD(MLP_ver1 epoch=5 lr=0.001)2.png>)
# 테스트 정확도
![test check](<./result/CDD(MLP_ver1 epoch=5 lr=0.001)accuracy.png>)

Epoch을 확인할 때랑 테스트를 해봤을 땐 생각보다 좋은 결과인 63%가 나와서 다른 파라미터들을 조금씩 바꿔가면서 성능을 끌어올려보고 싶었습니다.
# 심화 데이터 분석 (오차 행렬)
![confusion matrix test](<./result/CDD(MLP_ver1 epoch=5 lr=0.001)confusion matrix.png>)
![confusion matrix test](<./result/CDD(MLP_ver1 epoch=5 lr=0.001)confusion matrix2.png>)

그러나 심화 데이터를 확인하고 나니 AI가 판단을 해서 정답을 맞춘 게 아닌 사진 속 [[453,0],[262,0]] 를 보면 전부 불량품이라고 대답을 했고 
데이터셋 속 불량품 비율이 63% 였기 때문에 그러한 좋은 결과가 나온 것이었습니다.

심화 데이터를 확인하지 않았다면 몰랐을 정보였기 때문에 심화 데이터의 중요성을 알게 되었고 MLP의 한계도 알게 되었습니다.

