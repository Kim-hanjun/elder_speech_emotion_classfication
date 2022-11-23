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
## code formatter
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

## result
```
"klue/bert-base": {
    "eval_loss": 0.6249468922615051,
    "eval_accuracy": 0.7795758928571429,
    "eval_micro_f1": 0.7795758928571429,
    "eval_micro_recall": 0.7795758928571429,
    "eval_micro_precision": 0.7795758928571429,
    "eval_roc_auc_ovr": 0.9148944881136637,
    "eval_runtime": 13.6548,
    "eval_samples_per_second": 131.236,
    "eval_steps_per_second": 16.405,
    "epoch": 3.0
    }
```

```
"klue/roberta-base": {
"eval_loss": 0.6013084053993225,
"eval_accuracy": 0.7845982142857143,
"eval_micro_f1": 0.7845982142857143,
"eval_micro_recall": 0.7845982142857143,
"eval_micro_precision": 0.7845982142857143,
"eval_roc_auc_ovr": 0.9279247201889272,
"eval_runtime": 12.8652,
"eval_samples_per_second": 139.29,
"eval_steps_per_second": 17.411,
"epoch": 2.0
    }
    }
```
