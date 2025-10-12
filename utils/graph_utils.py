# D-Wave 양자 어닐러의 하드웨어 정보를 사용하도록 특별히 설계된 함수로,
# D-Wave QPU의 하드웨어 토폴로지 정보를 얻어와 로컬에 캐싱(caching)하는 역할
# 즉, D-Wave 그래프 생성 및 저장

import os
import pickle
import numpy as np
import random
from dwave.system import DWaveSampler, EmbeddingComposite

def load_or_create_graph(graph_path, n_visible, n_hidden, token=None):
    """
    Args:
        graph_path : 그래프 정보가 저장된 파일 경로
        n_visible : 가시 유닛의 수
        n_hidden : 은닉 유닛의 수
        token : D-Wave 토큰
    """

    # 그래프 정보가 이미 저장되어 있는 경우
    if os.path.exists(graph_path):
        with open(graph_path, 'rb') as f:
            graph_info = pickle.load(f)
        print("✅ Loaded saved D-Wave graph.")
    else:
        print("⚠️ No saved graph found. Creating new graph from D-Wave QPU.")
        sampler = EmbeddingComposite(DWaveSampler(solver={'name__contains': 'Advantage2'}, token=token))

        adjacency = sampler.child.adjacency
        # 이 코드는 선택된 D-Wave QPU의 물리적인 큐빗(qubit) 연결 구조(토폴로지) 정보를 가져옴
        # adjacency는 어떤 큐빗이 다른 어떤 큐빗과 물리적으로 연결되어 있는지를 나타내는 그래프


        # D-Wave 양자 컴퓨터의 물리적인 큐빗(qubit) 연결 구조(토폴로지)를 바탕으로, 
        # 우리가 사용할 RBM의 '연결 마스크(connection mask)'를 생성하는 부분
        all_qubits = list(adjacency.keys())
        random.shuffle(all_qubits)
        visible = all_qubits[:n_visible]
        hidden = all_qubits[n_visible:n_visible + n_hidden]
        mask = np.zeros((n_visible, n_hidden))
        for i, v in enumerate(visible):
            for j, h in enumerate(hidden):
                if h in adjacency[v]:
                    mask[i, j] = 1.0
        graph_info = {'adjacency': adjacency, 'visible': visible, 'hidden': hidden, 'mask': mask}
        with open(graph_path, 'wb') as f:
            pickle.dump(graph_info, f)
        print("✅ Saved new D-Wave graph.")
    return graph_info
