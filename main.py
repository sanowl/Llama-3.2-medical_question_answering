import os
import logging
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

def main():
    # Environment and Logging
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load Dataset
    dataset_path = ""
    dataset = load_from_disk(dataset_path)
    
    if "validation" not in dataset:
        split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({'train': split['train'], 'validation': split['test']})

    # Simplify dataset (use a smaller subset)
    dataset['train'] = dataset['train'].select(range(min(1000, len(dataset['train']))))
    dataset['validation'] = dataset['validation'].select(range(min(100, len(dataset['validation']))))

    # Load Model and Tokenizer
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Device Configuration (force CPU)
    device = torch.device("cpu")
    model.to(device)

    # Preprocessing Function
    def preprocess_function(examples):
        inputs = [f"{instruction} {input_text}" for instruction, input_text in zip(examples["instruction"], examples["input"])]
        targets = examples["output"]
        model_inputs = tokenizer(inputs, max_length=64, padding="max_length", truncation=True)
        model_targets = tokenizer(targets, max_length=64, padding="max_length", truncation=True)
        labels = [[(token if token != tokenizer.pad_token_id else -100) for token in label] for label in model_targets["input_ids"]]
        model_inputs["labels"] = labels
        return model_inputs

    # Apply Preprocessing
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    # Initialize Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        fp16=False,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=100,
        warmup_steps=100,
        learning_rate=3e-5,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"], 
        batch_size=training_args.per_device_train_batch_size, 
        shuffle=True,
        collate_fn=data_collator
    )
    
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        batch_size=1,
        collate_fn=data_collator
    )

    # Calculate Total Training Steps
    total_training_steps = (len(train_dataloader) // training_args.gradient_accumulation_steps) * training_args.num_train_epochs

    # Initialize Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=total_training_steps)

    # Training Loop
    def train_with_cot():
        model.train()
        global_step = 0
        for epoch in range(int(training_args.num_train_epochs)):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            for step, batch in enumerate(epoch_iterator):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / training_args.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                if global_step % training_args.logging_steps == 0:
                    logger.info(f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item() * training_args.gradient_accumulation_steps}")
                
                if training_args.eval_steps > 0 and global_step % training_args.eval_steps == 0:
                    evaluate(model, eval_dataloader, device, global_step)
                    model.train()
            
            save_checkpoint(model, tokenizer, epoch)
        
        save_final_model(model, tokenizer)

    # Evaluation Function
    def evaluate(model, dataloader, device, step):
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
        eval_loss /= len(dataloader)
        logger.info(f"Evaluation loss after step {step}: {eval_loss}")

    # Checkpoint Saving Function
    def save_checkpoint(model, tokenizer, epoch):
        checkpoint_dir = f"./results/checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Final Model Saving Function
    def save_final_model(model, tokenizer):
        final_model_dir = "./final_model"
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info(f"Training completed. Final model saved to {final_model_dir}")

    # Run Training
    train_with_cot()

    # Final Evaluation
    evaluate(model, eval_dataloader, device, "final")

if __name__ == "__main__":
    main()