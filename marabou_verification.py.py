# Marabou를 사용한 MNIST 분류기 검증
import numpy as np
import os
from maraboupy import Marabou
from maraboupy import MarabouCore
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

class MNISTMarabouVerifier:
    def __init__(self, onnx_model_path="mnist_mlp_small.onnx"):
        self.model_path = onnx_model_path
        self.network = None
        self.input_vars = None
        self.output_vars = None
        
    def load_model(self):
        """ONNX 모델을 Marabou에 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        print(f"ONNX 모델 로딩: {self.model_path}")
        self.network = Marabou.read_onnx(self.model_path)
        
        # 입력/출력 변수 가져오기
        self.input_vars = self.network.inputVars[0]  # 784차원 입력
        self.output_vars = self.network.outputVars[0]  # 10차원 출력
        
        print(f"입력 변수 수: {len(self.input_vars)}")
        print(f"출력 변수 수: {len(self.output_vars)}")
        
    def create_robustness_property(self, input_image, true_label, epsilon=0.1):
        """
        견고성(Robustness) 속성 생성
        입력 이미지에 epsilon 범위의 perturbation을 가해도 
        원래 분류 결과가 유지되는지 검증
        """
        # 입력 제약 조건 설정 (L∞ perturbation)
        for i, pixel_val in enumerate(input_image.flatten()):
            # 각 픽셀에 대해 [pixel_val - epsilon, pixel_val + epsilon] 범위 설정
            lower_bound = max(0.0, float(pixel_val - epsilon))  # 최소값 0
            upper_bound = min(1.0, float(pixel_val + epsilon))  # 최대값 1
            
            self.network.setLowerBound(self.input_vars[i], lower_bound)
            self.network.setUpperBound(self.input_vars[i], upper_bound)
        
        # 출력 제약 조건: 다른 모든 클래스보다 true_label의 출력이 더 큰지 검증
        # 즉, output[true_label] > output[j] for all j != true_label
        for j in range(10):
            if j != true_label:
                # output[true_label] - output[j] >= 0이어야 함
                # 이를 위반하는 경우(output[j] >= output[true_label])를 찾으면 
                # 반례(counterexample)가 존재함
                self.network.addInequality(
                    [self.output_vars[j], self.output_vars[true_label]], 
                    [1, -1], 
                    0
                )
    
    def create_local_robustness_property(self, input_image, epsilon=0.05):
        """
        지역 견고성 속성: 작은 perturbation에 대해 출력이 크게 변하지 않는지 검증
        """
        # 입력 제약 조건
        for i, pixel_val in enumerate(input_image.flatten()):
            lower_bound = max(0.0, float(pixel_val - epsilon))
            upper_bound = min(1.0, float(pixel_val + epsilon))
            
            self.network.setLowerBound(self.input_vars[i], lower_bound)
            self.network.setUpperBound(self.input_vars[i], upper_bound)
        
        # 원본 이미지의 예측 결과 계산 (PyTorch로)
        original_output = self.get_pytorch_prediction(input_image)
        max_class = np.argmax(original_output)
        
        print(f"원본 예측: 클래스 {max_class}, 신뢰도: {original_output[max_class]:.4f}")
        
        # 다른 클래스의 출력이 최대 클래스보다 크지 않아야 함
        for j in range(10):
            if j != max_class:
                self.network.addInequality(
                    [self.output_vars[j], self.output_vars[max_class]], 
                    [1, -1], 
                    0
                )
    
    def get_pytorch_prediction(self, input_image):
        """PyTorch 모델로 예측 (참고용)"""
        try:
            import torch
            import torch.nn as nn
            
            # 간단한 MLP 재정의 (위의 모델과 동일)
            class MNISTClassifier(nn.Module):
                def __init__(self):
                    super(MNISTClassifier, self).__init__()
                    self.flatten = nn.Flatten()
                    self.network = nn.Sequential(
                        nn.Linear(784, 16),
                        nn.ReLU(),
                        nn.Linear(16, 10)
                    )
                
                def forward(self, x):
                    x = self.flatten(x)
                    return self.network(x)
            
            # 더미 예측 (실제로는 훈련된 모델의 가중치를 로드해야 함)
            model = MNISTClassifier()
            model.eval()
            
            with torch.no_grad():
                if len(input_image.shape) == 2:
                    input_tensor = torch.tensor(input_image).unsqueeze(0).unsqueeze(0).float()
                else:
                    input_tensor = torch.tensor(input_image).unsqueeze(0).float()
                output = model(input_tensor)
                return torch.softmax(output, dim=1).numpy()[0]
        except:
            # PyTorch 사용 불가능한 경우 더미 출력
            return np.random.softmax(np.random.rand(10))
    
    def verify_property(self, timeout=300):
        """속성 검증 실행"""
        print("Marabou 검증 시작...")
        print("이 과정은 시간이 걸릴 수 있습니다...")
        
        # Marabou 옵션 설정
        options = Marabou.createOptions()
        options.timeoutInSeconds = timeout
        options.verbosity = 1
        
        # 검증 실행
        result = self.network.solve(options=options)
        
        return result
    
    def analyze_result(self, result):
        """검증 결과 분석"""
        if result[0] == "sat":
            print("속성이 위반되었습니다! (반례 발견)")
            print("반례(counterexample):")
            
            # 반례 입력 추출
            counterexample_input = []
            for var in self.input_vars:
                counterexample_input.append(result[1][var])
            
            counterexample_input = np.array(counterexample_input).reshape(28, 28)
            
            # 반례 출력 추출
            counterexample_output = []
            for var in self.output_vars:
                counterexample_output.append(result[1][var])
            
            print(f"반례 출력: {counterexample_output}")
            print(f"예측 클래스: {np.argmax(counterexample_output)}")
            
            return {
                'status': 'violation',
                'input': counterexample_input,
                'output': counterexample_output
            }
            
        elif result[0] == "unsat":
            print("속성이 만족됩니다! (견고함)")
            return {'status': 'verified'}
            
        else:
            print("⚠️ 검증 결과를 확정할 수 없습니다 (timeout 또는 unknown)")
            return {'status': 'unknown'}

def main():
    # 1. MNIST 테스트 이미지 로드
    print("MNIST 테스트 데이터 로딩...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # 첫 번째 테스트 이미지 선택
    test_image, true_label = test_dataset[0]
    test_image_np = test_image.numpy().squeeze()  # (28, 28)로 변환
    
    print(f"테스트 이미지 라벨: {true_label}")
    print(f"이미지 크기: {test_image_np.shape}")
    
    # 2. Marabou 검증기 초기화
    verifier = MNISTMarabouVerifier("mnist_mlp_small.onnx")
    
    try:
        # 3. 모델 로드
        verifier.load_model()
        
        # 4. 견고성 속성 생성
        print(f"\n견고성 검증 설정 중... (epsilon=0.1)")
        verifier.create_robustness_property(test_image_np, true_label, epsilon=0.1)
        
        # 5. 검증 실행
        result = verifier.verify_property(timeout=60)  # 1분 timeout
        
        # 6. 결과 분석
        analysis = verifier.analyze_result(result)
        
        # 7. 결과 시각화 (반례가 있는 경우)
        if analysis['status'] == 'violation':
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(test_image_np, cmap='gray')
            plt.title(f'원본 이미지\n라벨: {true_label}')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(analysis['input'], cmap='gray')
            plt.title(f'반례 이미지\n예측: {np.argmax(analysis["output"])}')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(analysis['input'] - test_image_np, cmap='RdBu')
            plt.title('차이 (perturbation)')
            plt.axis('off')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig('marabou_verification_result.png', dpi=150, bbox_inches='tight')
            plt.show()
            
    except FileNotFoundError:
        print("X ONNX 모델 파일이 없습니다.")
        print("먼저 첫 번째 스크립트를 실행하여 모델을 생성하세요.")
    except Exception as e:
        print(f"X 검증 중 오류 발생: {str(e)}")
        print("Marabou가 올바르게 설치되었는지 확인하세요.")

if __name__ == "__main__":
    main()