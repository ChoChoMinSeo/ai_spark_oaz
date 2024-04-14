# ai_spark_oaz

## 1. 프로젝트 개요

### 프로젝트 주제

### **글로벌 산불 감지 챌린지🌋 :  AI를 활용한 향상된 위성 이미지 분석**

위성 데이터를 활용한 실시간 산불 감시 및 조기 탐지를 통해 산불 재난을 모니터링하고 예측하는 솔루션을 개발하는 문제입니다

위성 사진

![example image](./assets/Untitled%203.png)

### 프로젝트 요약

- 재현성 검증을 위해 Seed 고정 및 학습한 가중치 관리의 필요성
- 제공된 코드에서 기능적인 부분을 개선 및 추가하여 자체적 Baseline 구축
- 관련 논문 및 EDA 용 python 코드를 통한 dataset 이해
- 효과적인 augmentation 및 loss function 선정 및 custom

### 협업 툴 링크

노션 페이지 [6th 2024 AI spark challenge](https://www.notion.so/6th-2024-AI-spark-challenge-d3e9941a89f245a680d783cc4a82ac43?pvs=21) 

### 프로젝트 목표

- 논리적인 모델 선정 및 데이터 처리 방법을 습득
- 대회의 흐름을 파악하고 목적을 달성해 리더보드 높은 순위 기록
- 데이터 분석 능력 및 전처리 역량 획득
- Github, Notion 등 협업 툴 고급 활용 및 협업과 소통 능력 함양

### 프로젝트 구조

```
ai_spark_oaz/
│
├── baseline_ver1.3.ipynb - train, inference and make submission code
├── baseline_ver1.3CW.ipynb - train, inference and make submission code for CWUnet
│
├── experiments/
|   ├── eda.ipynb
|   └── build_sub_trainset.ipynb
│
├── baselines/ - base baseline and baseline code by version
│   ├── 제6회_2024_연구개발특구_AI_SPARK_챌린지_baseline.ipynb
│   ├── baseline_ver1.0.ipynb
│   ├── baseline_ver1.1.ipynb
│   └── baseline_ver1.2.ipynb
│
├── models/ - FCN, CWUnet_v1, CWUnet_v2
│   ├── fcn_model.py
│   ├── CWUnet.py
│   └── CWUnet_2.py
│
└── utils/ - ensemble, defined loss functions
    ├── ensemble.ipynb
    └── loss_fn.py
```

## 2. 프로젝트 구성 및 역할 - 각자

- 프로젝트 전반: Baseline 작성, EDA, 모델 실험
- 프로젝트 후반: 전처리, 채널 조합 실험
- 역할
    - 조민서: Baseline 코드 배포, CWNet 코드 구현, PM 역할 수행
    - 박지우: EDA, Dataset 구성

## 3. 프로젝트 수행 결과
