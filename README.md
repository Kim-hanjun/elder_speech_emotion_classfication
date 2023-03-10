# elder_speech_emotion_classfication

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
    "eval_loss": 0.9755731225013733,
    "eval_accuracy": 0.8052455357142857,
    "eval_micro_f1": 0.8052455357142857,
    "eval_micro_recall": 0.8052455357142857,
    "eval_micro_precision": 0.8052455357142857,
    "eval_roc_auc_ovr": 0.9223152089334774,
    "eval_runtime": 12.9615,
    "eval_samples_per_second": 138.256,
    "eval_steps_per_second": 17.282,
    "epoch": 4.0
    }
```

```
"monologg/koelectra-base-v3-discriminator": {
    "eval_loss": 0.9988111853599548,
    "eval_accuracy": 0.7896205357142857,
    "eval_micro_f1": 0.7896205357142857,
    "eval_micro_recall": 0.7896205357142857,
    "eval_micro_precision": 0.7896205357142857,
    "eval_roc_auc_ovr": 0.8968455773267365,
    "eval_runtime": 13.2091,
    "eval_samples_per_second": 135.664,
    "eval_steps_per_second": 16.958,
    "epoch": 6.0
    }
```
