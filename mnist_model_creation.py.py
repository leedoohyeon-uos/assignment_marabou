# MNIST MLP 모델 생성 및 ONNX 변환
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 간단한 MLP 모델 정의
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(784, 16),  # 입력층: 28*28=784 -> 16
            nn.ReLU(),
            nn.Linear(16, 10)    # 출력층: 16 -> 10 (클래스 수)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# 2. 모델 훈련 (간단한 버전)
def train_model():
    # 데이터 로딩
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 모델, 손실함수, 최적화기 설정
    model = MNISTClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 훈련 (간단히 1 에포크만)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 500:  # 빠른 훈련을 위해 제한
            break
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return model

# 3. ONNX로 변환
def convert_to_onnx(model, output_path="mnist_mlp_small.onnx"):
    model.eval()
    
    # 더미 입력 생성 (배치 크기 1, 1채널, 28x28)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # ONNX로 변환
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"모델이 {output_path}로 저장되었습니다.")

# 4. 모델 테스트
def test_model(model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'테스트 정확도: {accuracy:.2f}%')
    return accuracy

# 메인 실행 함수
def main():
    print("MNIST MLP 모델 생성 및 훈련 시작...")
    
    # 모델 훈련
    model = train_model()
    
    # 모델 테스트
    accuracy = test_model(model)
    
    # ONNX로 변환
    convert_to_onnx(model)
    
    # 모델 구조 출력
    print("\n모델 구조:")
    print(model)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params}")
    
    return model

if __name__ == "__main__":
    model = main()