#실행 코드 : python main.py --config ./config.py

import argparse
import importlib.util

def load_config_py(path):
    """
    Python 파일을 로드하여 설정 정보를 반환하는 함수

        importlib.util은 파일 경로에서 직접 모듈을 로드하는 기능을 제공 
        아래 단계대로 spec → module → exec 순서로 실행
    
        1.	spec_from_file_location(name, path)
	        •	모듈의 “메타데이터(specification)”를 생성
	        •	name은 임시로 붙이는 모듈 이름, path는 실제 파일 경로
	    2.	module_from_spec(spec)
	        •	spec을 이용해 아직 “실행 전”인 모듈 객체를 생성
	    3.	sys.modules[...] = module
	        •	파이썬의 전역 모듈 캐시에 등록하면, 이후 import my_module로도 접근 가능
	    4.	spec.loader.exec_module(module)
	        •	모듈 소스 파일을 읽어서 실행 → 이제 모듈 안의 함수/클래스가 사용 가능

    Args:
        path (str): Python 파일의 경로

    Returns:
        dict: 설정 정보
    """
    spec = importlib.util.spec_from_file_location("config_module", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_config()

# 실행 코드 : python main.py --config /Users/joeycho/Desktop/pgm/202509_RBM/DQA4RBM_Re/config.py
# ★ main.py랑 config.py가 같은 폴더에 있으면: python main.py --config ./config.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Python config file")
    args = parser.parse_args()

    config = load_config_py(args.config)
    print(config)

    from trainers.trainer import train  
    train(config)