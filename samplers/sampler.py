# samplers/sampler.py
import torch
import numpy as np
import time

# D-Wave 모드에서 사용되는 모듈
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite

# 전역 샘플러 캐시 (재사용을 위해)
_sampler_cache = None

def reset_sampler_cache():
    """샘플러 캐시를 초기화하는 함수 (연결 문제 해결용)"""
    global _sampler_cache
    _sampler_cache = None
    print("🔄 Sampler cache reset.")

def sample_negative_particles(mode, rbm, config, mask_tensor, graph_info=None, persistent_v=None):
    """
    Input:
        mode
            : 'pcd' or 'dwave'
        rbm
            : RBM model
        config
            : configuration
        mask_tensor
            : mask tensor => RBM에서 연결 그래프를 제한하는 마스크 (sparse connectivity용)
        graph_info
            : graph information => D-Wave 임베딩 시 필요한 그래프 구조 정보
        persistent_v
            : persistent visible units => PCD에서 persistent chain을 유지할 때 쓰이는 visible 상태
    Output:
        v_model (torch.Tensor)
            : visible units
        h_model (torch.Tensor)
            : hidden units
    """

    device = torch.device('cpu')

    # PCD 모드일 경우
    if mode == 'pcd':
        all_samples = []

        for _ in range(config['num_reads']):
            with torch.no_grad():
                for _ in range(config['k']):
                    h_persistent = rbm.sample_h(persistent_v)
                    persistent_v = rbm.sample_v(h_persistent)
            all_samples.append(persistent_v[0].detach().clone())

        # visible units size: (num_reads, n_visible)
        v_model = torch.stack(all_samples).to(device)

        # Z2 symmetry fix
        # 각 샘플에서 첫 번째 비트가 –1이면, 전체 벡터를 뒤집어 항상 첫 번째 비트가 +1이 되도록 정렬(alignment)
        for i in range(v_model.size(0)):
            if v_model[i, 0] == -1:
                v_model[i] *= -1     # 벡터의 모든 원소에 -1을 곱하는 것 → 벡터 전체가 부호 반전 

        h_model = torch.tanh(v_model @ (rbm.W * mask_tensor))
        h_model = torch.sign(h_model + torch.randn_like(h_model) * 0.01)

        return v_model, h_model, persistent_v

    # D-Wave 모드일 경우
    elif mode == 'dwave':
        W_np = (rbm.W * mask_tensor).detach().cpu().numpy() # W (n_visible, n_hidden)

        # Ising 모델에서의 local field (편향 항)을 의미
        h = {i: 0.0 for i in range(config['n_visible'] + config['n_hidden'])}

        # Ising 모델에서의 coupling terms (상호작용 항)을 의미
        J = {(i, config['n_visible'] + j): -W_np[i, j]/config['beta_rescale']
             for i in range(config['n_visible'])
             for j in range(config['n_hidden']) if graph_info['mask'][i, j] != 0}

        bqm = BinaryQuadraticModel.from_ising(h, J)
        # BinaryQuadraticModel.from_ising(h, J) : Ising model을 BinaryQuadraticModel로 변환
        # h : bias terms
        # J : coupling terms

        # D-Wave 샘플러 설정 => D-Wave 하드웨어에 접속할 준비 (재사용)
        # (수정) 
        global _sampler_cache
        if _sampler_cache is None:
            print("🔗 Creating D-Wave sampler connection...")
            _sampler_cache = EmbeddingComposite(DWaveSampler(token=config.get('token')))
            print("✅ D-Wave sampler connection established.")
        sampler = _sampler_cache
        # EmbeddingComposite: RBM의 그래프 구조(visible–hidden bipartite graph)를 하드웨어의 체인 구조에 자동으로 임베딩할 때 필요한 모듈
        # DWaveSampler: D-Wave 시스템에 연결하고 문제를 제출하는 데 사용되는 클래스

        # (시간 측정 부분 추가) =================================================
        # D-Wave 샘플러를 사용하여 샘플링 수행
        print(f"🚀 Starting D-Wave sampling with {config['num_reads']} reads...")
        
        # 상세한 시간 측정
        bqm_time = time.time()
        print(f"📊 BQM created with {len(J)} couplings")
        
        request_time = time.time()
        
        # 임베딩 실패 시 재시도 로직
        max_retries = 3
        for attempt in range(max_retries):
            try:
                sampleset = sampler.sample(
                    bqm,
                    num_reads=config['num_reads'],
                    auto_scale=False,
                    fast_anneal=True,   # 빠른 어닐링 모드 사용
                    anneal_schedule=config['anneal_schedule'], # 맞춤형 어닐링 스케줄 (시간–세기 곡선)
                    label="RBM training"
                )
                break  # 성공하면 루프 탈출
            except ValueError as e:
                if "no embedding found" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Embedding failed (attempt {attempt + 1}/{max_retries}). Retrying with new sampler...")
                    reset_sampler_cache()  # 캐시 초기화
                    sampler = EmbeddingComposite(DWaveSampler(token=config.get('token')))
                    _sampler_cache = sampler  # 새 샘플러로 캐시 업데이트
                else:
                    raise e  # 최대 재시도 횟수 초과 시 오류 발생
        total_time = time.time() - request_time
        
        # D-Wave 응답 정보 분석
        if hasattr(sampleset, 'info'):
            timing_info = sampleset.info.get('timing', {})
            qpu_time_us = timing_info.get('qpu_access_time', 0)  # 마이크로초 단위
            qpu_time = qpu_time_us / 1_000_000  # 마이크로초를 초로 변환
            network_time = total_time - qpu_time
            
            print(f"⏱️ Total time: {total_time:.2f}s")
            print(f"🔬 QPU time: {qpu_time:.2f}s ({qpu_time_us}μs)")
            print(f"🌐 Network time: {network_time:.2f}s")
            print(f"📈 Energy range: {sampleset.record.energy.min():.2f} to {sampleset.record.energy.max():.2f}")
        else:
            print(f"⏱️ D-Wave sampling completed in {total_time:.2f}s")
        # D-Wave 하드웨어에서 num_reads개의 spin configuration을 샘플링하여 반환

        #=================================================

        samples = sampleset.record.sample.astype(np.int8)
        samples = 2 * samples - 1 # D-Wave 결과는 기본적으로 {0,1} 값으로 샘플링 -> {-1, +1} 스핀으로 변환

        # Z2 symmetry fix
        for i in range(samples.shape[0]):
            if samples[i, 0] == -1:
                samples[i] *= -1

        # samples size : (num_reads, n_visible + n_hidden)
        v_model = samples[:, :config['n_visible']]
        h_model = samples[:, config['n_visible']:]

        # Torch 텐서로 변환
        v_model = torch.tensor(v_model, dtype=torch.float32, device=device)
        h_model = torch.tensor(h_model, dtype=torch.float32, device=device)

        return v_model, h_model, None

    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
