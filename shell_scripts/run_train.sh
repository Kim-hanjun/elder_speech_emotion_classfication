NUM_GPU=3
GPU_IDS="1,2,3"
export OMP_NUM_THREADS=8
model_name_or_path="klue/roberta-base"
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU train.py \
  --output_dir "output/${model_name_or_path}" \
  --train_csv_path "data/preprocess/train.csv" \
  --test_csv_path "data/preprocess/test.csv" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1\
  --model_name_or_path ${model_name_or_path} \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --logging_strategy "steps" \
  --logging_steps 50 \
  --eval_step 100 \
  --save_step 100 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --logging_strategy "steps" \
  --load_best_model_at_end \
  --metric_for_best_model "micro_f1" \
  --learning_rate 2e-5 \
  --dataloader_num_workers "4" \
  --label_names "labels" \
  --do_eval \
  --fp16