# utils/metrics.py
# traning error & validation error의 metric으로 hamming distance with symmetry 계산

import torch

# 두 벡터 x와 y의 hamming 거리를 계산하는 함수
# 길이가 같은 두 문자열(또는 비트열)에서 서로 다른 문자의 개수를 세어, 두 데이터가 얼마나 다른지를 측정
# => 이 함수는 y 자체와의 거리와 y를 완전히 뒤집은(-y) 상태와의 거리를 모두 계산한 후, 둘 중 더 짧은 거리를 선택
def hamming_distance_with_symmetry(x, y):
    dist_direct = (x != y).sum(dim=1)
    dist_flipped = (x != -y).sum(dim=1)
    return torch.min(dist_direct, dist_flipped).float() / x.size(1)  # 0과 1 사이의 비율