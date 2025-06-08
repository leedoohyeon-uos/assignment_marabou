# Marabou를 사용한 MNIST 분류기 검증 프로젝트

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)
![Marabou](https://img.shields.io/badge/Marabou-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

##  프로젝트 목표

- MNIST 분류기의 적대적 견고성 형식적 검증
- 작은 입력 변화에 대한 모델 안정성 분석
- 적대적 예시(adversarial examples) 자동 생성
- 신경망 검증 도구 Marabou의 실무 적용 방법 학습

### 모델 구조

```python
MNISTClassifier(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (network): Sequential(
    (0): Linear(in_features=784, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=10, bias=True)
  )
)
```

- **입력**: 28×28 MNIST 이미지 (784개 픽셀)
- **은닉층**: 16개 뉴런 (ReLU 활성화)
- **출력**: 10개 클래스 (0-9 숫자)
- **총 파라미터**: 12,810개

### 검증 속성

#### 견고성 속성 (Robustness Property)

수학적 정의:
```
∀x. ||x - x₀||∞ ≤ ε → argmax(f(x)) = argmax(f(x₀))
```

구현:
```python
# L∞ perturbation 제약
for i, pixel_val in enumerate(input_image.flatten()):
    lower_bound = max(0.0, float(pixel_val - epsilon))
    upper_bound = min(1.0, float(pixel_val + epsilon))
    network.setLowerBound(input_vars[i], lower_bound)
    network.setUpperBound(input_vars[i], upper_bound)

# 분류 조건 제약
for j in range(10):
    if j != true_label:
        network.addInequality(
            [output_vars[j], output_vars[true_label]], 
            [1, -1], 
            0
        )
```

### 파라미터 조정

#### Epsilon 값 변경

```python
# 더 엄격한 검증
verifier.create_robustness_property(test_image_np, true_label, epsilon=0.01)

# 더 관대한 검증
verifier.create_robustness_property(test_image_np, true_label, epsilon=0.3)
```

#### 타임아웃 설정

```python
# 더 긴 검증 시간
result = verifier.verify_property(timeout=600)  # 10분
```

#### 다른 이미지 테스트

```python
# 특정 인덱스의 이미지 사용
test_image, true_label = test_dataset[42]  # 42번째 이미지

# 특정 클래스의 이미지 찾기
for i, (image, label) in enumerate(test_dataset):
    if label == 3:  # 숫자 3인 이미지
        test_image, true_label = image, label
        break
```