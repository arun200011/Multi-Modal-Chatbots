# training/train_text.py
import json, os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--faq", default="../data/faq.json")
parser.add_argument("--out_dir", default="../models/text_qa")
parser.add_argument("--model_name", default="distilbert-base-uncased-distilled-squad")
parser.add_argument("--epochs", type=int, default=2)
args = parser.parse_args()

# load faq pairs
with open(args.faq) as f:
    faq = json.load(f)

# build small SQuAD-like dataset: each QA becomes context=answer, question=question? Better: context could be combined domain doc; for simplicity put answer as context so model can extract it.
examples = {"id": [], "title": [], "context": [], "question": [], "answers": []}
for i, qa in enumerate(faq):
    q = qa.get("question")
    a = qa.get("answer")
    examples["id"].append(str(i))
    examples["title"].append("faq")
    examples["context"].append(a)
    examples["question"].append(q)
    answers = {"text": [a], "answer_start": [0]}
    examples["answers"].append(answers)

ds = Dataset.from_dict(examples)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def preprocess_function(examples):
    inputs = tokenizer(examples["question"], examples["context"], truncation=True, padding="max_length", max_length=256)
    # find start/end positions
    start_positions = []
    end_positions = []
    for i, ans in enumerate(examples["answers"]):
        start = examples["context"][i].find(ans["text"][0])
        if start == -1:
            start_positions.append(0)
            end_positions.append(0)
        else:
            end = start + len(ans["text"][0])
            token_start_index = inputs.char_to_token(i, start)
            token_end_index = inputs.char_to_token(i, end-1)
            if token_start_index is None:
                token_start_index = 0
            if token_end_index is None:
                token_end_index = 0
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized = ds.map(preprocess_function, batched=True, remove_columns=ds.column_names)

model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
training_args = TrainingArguments(
    output_dir=args.out_dir,
    evaluation_strategy="no",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=8,
    save_total_limit=1,
    logging_steps=5
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=default_data_collator,
    tokenizer=tokenizer
)
trainer.train()
trainer.save_model(args.out_dir)
print("Saved text QA model to", args.out_dir)
