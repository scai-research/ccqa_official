import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed
)
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# 재현성을 위한 시드 설정
set_seed(42)

# 1. 데이터 로드( 직접 만든 데이터셋 사용 )
with open("/data3/jykim/Projects/CCQA_official/finetuning/combined_qa_dataset.json", "r") as f:
    raw_data = json.load(f)

# 데이터 검증 - NaN 값이나 문제가 있는 샘플 필터링
valid_data = []
for item in raw_data:
    if isinstance(item.get('input'), str) and isinstance(item.get('output'), str):
        if len(item['input'].strip()) > 0 and len(item['output'].strip()) > 0:
            valid_data.append(item)
print(f"Valid data: {len(valid_data)}/{len(raw_data)} ({len(valid_data)/len(raw_data)*100:.2f}%)")

# 데이터를 학습용과 평가용으로 분할 (90% 학습, 10% 평가)
train_data, eval_data = train_test_split(valid_data, test_size=0.1, random_state=42)

# Dataset 객체 생성
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# 2. 모델 및 토크나이저 불러오기
model_name = "google/flan-t5-base"  # flan-t5-base 모델 사용
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. 전처리
def preprocess(example):
    # 데이터셋에 따라 다른 프롬프트 형식 적용
    dataset_type = example.get("dataset", "").strip()
    
    if dataset_type == "GSM8K" or dataset_type == "SVAMP":
        # GSM8K와 SVAMP 모두 동일한 프롬프트 사용
        prompt = f"CRITICAL: Do not change ANY numeric values in the answer. Every number (59, 8, 74, etc.) must be preserved EXACTLY in your question. Generate a question that would have this as its answer: {example['input']}"
    
    elif dataset_type == "CommonSenseQA":
        # CommonSenseQA 데이터셋용 프롬프트 (선택지 없이 질문만 생성)
        prompt = f"CRITICAL: From the commonsense reasoning answer provided below, recreate the original commonsense reasoning question. No need to include choices. Answer: {example['input']}"
    
    else:
        # 기본 프롬프트 형식 (데이터셋이 명시되지 않은 경우)
        prompt = f"Generate the original detailed question based on the given answer: {example['input']}"
    
    target = example["output"]
    
    inputs = tokenizer(
        prompt,
        max_length=256,  
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    targets = tokenizer(
        target,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # T5 모델의 라벨링
    labels = targets["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # 패딩 토큰을 -100으로 마스킹
    
    return {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "labels": labels[0]
    }

# 학습 및 평가 데이터셋 전처리
tokenized_train_dataset = train_dataset.map(
    preprocess,
    remove_columns=["input", "output", "dataset"],
    batched=False
)

tokenized_eval_dataset = eval_dataset.map(
    preprocess,
    remove_columns=["input", "output", "dataset"],
    batched=False
)

# 4. 학습 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-fullfinetuned",
    per_device_train_batch_size=8,     
    per_device_eval_batch_size=8,      
    gradient_accumulation_steps=4,     
    learning_rate=5e-5,                
    num_train_epochs=5,                
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    max_grad_norm=1.0,                
    fp16=False,                        
    report_to="none",
    weight_decay=0.01,
    lr_scheduler_type="cosine",        
    save_total_limit=3,
    predict_with_generate=True,
    generation_max_length=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    gradient_checkpointing=True,       
)

# 5. Trainer 구성
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
)

# 6. 학습 시작
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    # 모델을 저장하여 지금까지의 학습을 보존
    model.save_pretrained("/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-checkpoint")
    tokenizer.save_pretrained("/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-checkpoint")
    print("Saved checkpoint of the model before crash")
    raise

# 7. 저장
model.save_pretrained("/data3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-fullfinetuned")
tokenizer.save_pretrained("/dat3/jykim/Projects/CCQA_official/Finetuned_models/flan-t5-base-fullfinetuned")