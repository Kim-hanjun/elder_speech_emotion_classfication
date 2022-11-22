# elder_spech_emotion_classfication

## environment
```
os==Ubuntu 18.04.6 LTS
python==3.8
cuda=11.0
```
## install
```
python3.8 -m venv .venv
pip install --upgrade pip
pip install -r requirements.txt
bash shell_scripts/install_torch.sh
```
## code formater
```
black --line-length 119
isort
```

## preprocess
```
bash shell_scripts/run_preprocess_raw_data.sh
```
## train
```
bash shell_scripts/run_train.sh
```

## log
- log/preprocess.log : 데이터 전처리 로그
- YYYYMMDD-HH:MM:SS_result.json : 모델 best checkpoint의 테스트 결과
