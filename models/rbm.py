import torch

class RBM(torch.nn.Module):
    """
    RBM Model
        - 오직 W (가중치)만 있음 => visible bias (b)와 hidden bias (c)는 생략
        - 여기서는 torch.tanh + torch.sign으로 근사했는데, 
          이는 hidden/visible을 {-1, +1} 스핀으로 취급한 이진 변수 버전(이른바 ±1 RBM)에 가까움
    """

    def __init__(self, n_vis, n_hid, mask):
        """
        Args:
            n_vis : number of visible units
            n_hid : number of hidden units
            mask : mask matrix
        """
        super().__init__()
        # 여타 initialization에 비해 이와 같은 방식이 학습이 제일 잘 됨
        self.W = torch.nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.mask = mask

    def sample_h(self, v):
        """
        주어진 visible units로부터 hidden unit을 sampling하는 과정을 모델링하는 함수
        (batch, n_visible) @ (n_visible, n_hidden) → (batch, n_hidden)
        
        Args:
            v : visible units
        """
        p = torch.tanh(v @ (self.W * self.mask)) 
        # @ : matrix multiplication, * : element-wise multiplication
        return torch.sign(p + torch.randn_like(p) * 0.01)
        # torch.randn_like(tensor) = torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device)


    def sample_v(self, h):
        """
        주어진 hidden units로부터 visible unit을 sampling하는 과정을 모델링하는 함수
        (batch, n_hidden) @ (n_hidden, n_visible) → (batch, n_visible)
        Args:
            h : hidden units
        """
        p = torch.tanh(h @ (self.W * self.mask).t())
        return torch.sign(p + torch.randn_like(p) * 0.01)