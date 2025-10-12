# utils/plot_utils.py
# 학습된 RBM이 주어진 입력을 얼마나 잘 '이해'하고 '복원'하는지 눈으로 확인하는 과정
# RBM의 생성 능력을 간접적으로 평가하는 지표로 활용

import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def save_reconstruction_images(rbm, data_loader, n_visible, save_path, device='cpu'):
    with torch.no_grad():
        for data, _ in data_loader:
            v_data = data.view(-1, 784)[:, :n_visible].to(device)
            v_data = torch.sign(v_data * 2 - 1)  # RBM이 처리하는 데이터 형식을 {0, 1}이 아닌 {-1, 1}로 변환
            h_sample = rbm.sample_h(v_data)
            v_recon = rbm.sample_v(h_sample)
            break  # break 때문에 루프는 한 번만 실행 -> 첫 번째 batch만 사용

    v_recon = (v_recon + 1) / 2.0  # RBM이 출력한 {-1, 1} 상태의 재구성 데이터를 다시 0~1 범위의 이미지 픽셀 값으로 변환

    # 재구성 데이터를 이미지 형태로 변환
    v_recon_images = v_recon.view(-1, 1, 28, 28).cpu()
    v_recon_subset = v_recon_images[:100]

    # vutils.make_grid : 여러 이미지를 하나의 그리드 형태로 결합하는 함수
    # nrow=10 : 한 행에 10개의 이미지를 배치
    # padding=2 : 이미지 간의 간격
    # normalize=True : 이미지 픽셀 값을 0~1 범위로 정규화
    grid = vutils.make_grid(v_recon_subset, nrow=10, padding=2, normalize=True)

    # 이미지 저장
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title('Final Reconstruction Samples (Best Model)')
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Reconstruction image saved to {save_path}")