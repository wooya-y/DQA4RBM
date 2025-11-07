def get_config():

    # === 공통 설정 ===
    mode = 'dwave'  # 'pcd' 또는 'dwave' 중 택 1
    dataset = 'MNIST'
    n_visible = 784
    n_hidden = 1200
    epochs = 20
    batch_size = 50000
    initial_lr = 0.005
    beta_rescale = 1.0
    num_reads = 3000   # Negative Phase 계산을 위한 Sample 횟수

    base_config = {
        'mode': mode,             # 'pcd' 또는 'dwave' 중 택 1
        'dataset': dataset,       # 'MNIST', 'fMNIST', 'kMNIST' 중 택 1
        'n_visible': n_visible,   # 가시 유닛의 수
        'n_hidden': n_hidden,     # 은닉 유닛의 수
        'epochs': epochs,         # 에포크 횟수
        'batch_size': batch_size, # 배치 크기
        'initial_lr': initial_lr, # 초기 학습률
        'save_dir': f'./results/{dataset}{n_hidden}',
        'graph_path': f'./results/dwave_graph_{n_hidden}.pkl',
        'l2_weight_decay': 0
    }

    # === PCD 모드일 경우 ===
    if mode == 'pcd':
        k = 100
        base_config.update({
            'k': k,
            'num_reads': num_reads,
            'save_dir': f"{base_config['save_dir']}/pcd{k}",
        })

    # === D-Wave 모드일 경우 ===
    elif mode == 'dwave':
        
        anneal_schedule = [(0.0, 0.0), (0.005, 1.0)]

        # D-Wave token 설정 (실제 토큰으로 교체하세요)
        dwave_token = ""  # 여기에 실제 D-Wave token을 입력하세요
        
        base_config.update({
            'beta_rescale': beta_rescale,
            'anneal_schedule': anneal_schedule,
            'num_reads': num_reads,
            'save_dir': f"{base_config['save_dir']}/dwave{beta_rescale}",
            'token': dwave_token,  # D-Wave token 추가
        })

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    return base_config
