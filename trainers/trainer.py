# trainers/trainer.py

import time
import torch
import pickle
import os
from models.rbm import RBM
from samplers.sampler import sample_negative_particles

from utils.metrics import hamming_distance_with_symmetry
from utils.graph_utils import load_or_create_graph
from utils.data_utils import load_data
from utils.plot_utils import save_reconstruction_images


def train(config):
    """
    Args:
        config : 설정 정보
    """
    device = torch.device('cpu')
    mode = config['mode']
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # --- Load or create graph ---
    graph_info = load_or_create_graph(
        config['graph_path'],
        config['n_visible'],
        config['n_hidden'],
        token=config.get('token')
    )
    mask_tensor = torch.tensor(graph_info['mask'], dtype=torch.float32).to(device)

    # --- Data ---
    train_loader, val_loader = load_data(config['dataset'], config['n_visible'])

    # --- Model ---
    rbm = RBM(config['n_visible'], config['n_hidden'], mask_tensor).to(device)

    # --- Optimizer: L2 only (weight decay) ---
    opt = torch.optim.SGD(
        rbm.parameters(),
        lr=config.get('initial_lr', 0.01),
        weight_decay=config.get('l2_weight_decay', 0.0)  # <= L2 strength (e.g., 1e-4)
    )

    # --- Persistent chain (for PCD only) ---
    persistent_v = torch.sign(torch.randn(1, config['n_visible'])).to(device) if mode == 'pcd' else None

    # --- Epoch 0 reconstruction error 측정 (시작 시 오차 확인) ---
    # train/val 데이터 한 배치만 뽑아 재구성 에러(hamming distance with symmetry) 계산 -> baseline log
    train_error_list, val_error_list, time_epoch = [], [], []
    best_val_error, best_model_state = float('inf'), None

    with torch.no_grad():
        for data, _ in train_loader:
            v0 = data.view(-1, config['n_visible']).to(device)
            v0 = torch.sign(v0 * 2 - 1)
            h0 = rbm.sample_h(v0)
            v0_recon = rbm.sample_v(h0)
            train_error = hamming_distance_with_symmetry(v0, v0_recon).mean().item()
            break

        for data, _ in val_loader:
            v_val = data.view(-1, config['n_visible']).to(device)
            v_val = torch.sign(v_val * 2 - 1)
            h_val = rbm.sample_h(v_val)
            v_val_recon = rbm.sample_v(h_val)
            val_error = hamming_distance_with_symmetry(v_val, v_val_recon).mean().item()
            break

    print(f"[{mode.upper()}] Epoch 0 | Train Error: {train_error:.4f} | Val Error: {val_error:.4f}")
    train_error_list.append(train_error)
    val_error_list.append(val_error)

    # --- Training Loop ---
    for epoch in range(config['epochs']):
        start_time = time.time()

        # (Optional) simple LR schedule hook
        for param_group in opt.param_groups:
            param_group['lr'] = config.get('initial_lr', 0.01)

        # One batch only (full-batch)
        for data, _ in train_loader:
            v_data = data.view(-1, config['n_visible']).to(device)
            v_data = torch.sign(v_data * 2 - 1)   # in {−1, +1}
            h_data = rbm.sample_h(v_data)
            break

        # Negative sampling (PCD or D-Wave)
        v_model, h_model, persistent_v = sample_negative_particles(
            mode, rbm, config, mask_tensor, graph_info, persistent_v
        )

        # --- Contrastive divergence gradient (masked) ---
        pos_grad = v_data.T @ h_data / v_data.size(0)     # (V,H)
        neg_grad = v_model.T @ h_model / v_model.size(0)  # (V,H)
        rbm.W.grad = -(pos_grad - neg_grad) * mask_tensor
        
        # --- L2 is handled by optimizer's weight_decay automatically ---
        opt.step()
        opt.zero_grad()

        # --- Evaluation (train/val reconstruction error) ---
        h_sample = rbm.sample_h(v_data)
        v_recon = rbm.sample_v(h_sample)
        train_error = hamming_distance_with_symmetry(v_data, v_recon).mean().item() # humming distance 기록 

        with torch.no_grad():
            for data, _ in val_loader:
                v_val = data.view(-1, config['n_visible']).to(device)
                v_val = torch.sign(v_val * 2 - 1)
                h_val = rbm.sample_h(v_val)
                v_val_recon = rbm.sample_v(h_val)
                val_error = hamming_distance_with_symmetry(v_val, v_val_recon).mean().item() # humming distance 기록 
                break

        elapsed = time.time() - start_time
        print(f"[{mode.upper()}] Epoch {epoch+1} | Train Error: {train_error:.4f} | Val Error: {val_error:.4f} | Time: {elapsed:.2f}s")

        train_error_list.append(train_error)
        val_error_list.append(val_error)
        time_epoch.append(elapsed)

        # best model 저장
        if val_error < best_val_error:
            best_val_error = val_error
            best_model_state = rbm.state_dict()

    # --- Save best model & logs ---
    rbm.load_state_dict(best_model_state)
    graph_info['model_state_dict'] = best_model_state
    with open(os.path.join(save_dir, 'final_model_graph.pkl'), 'wb') as f:
        pickle.dump(graph_info, f)
    with open(os.path.join(save_dir, 'errors.pkl'), 'wb') as f:
        pickle.dump([train_error_list, val_error_list, time_epoch], f)

    save_reconstruction_images(
        rbm,
        train_loader,
        n_visible=config['n_visible'],
        save_path=os.path.join(save_dir, "reconstruction_best.png"),
        device=device
    )
