import os
import logging
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import psutil  # For memory monitoring

def main():
    # ============================================
    # 1. Environment and Logging Setup
    # ============================================
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer parallelism warnings

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # ============================================
    # 2. Memory Monitoring Function
    # ============================================
    def log_memory_usage(stage):
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 3)  # in GB
        logger.info(f"Memory usage at {stage}: {mem:.2f} GB")

    # ============================================
    # 3. Device Configuration
    # ============================================
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(device) else torch.float16  # Use BF16 if supported
        logger.info("Using CUDA device for training.")
    else:
        raise RuntimeError("CUDA device not available. Please ensure you have a compatible NVIDIA GPU.")

    log_memory_usage("after device setup")

    # ============================================
    # 4. Load and Prepare Dataset
    # ============================================
    dataset_path = "/path/to/your/medical_question_answering_dataset"  # Replace with your dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    dataset = load_from_disk(dataset_path)
    logger.info("Dataset loaded from disk.")

    if "validation" not in dataset:
        split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({'train': split['train'], 'validation': split['test']})
        logger.info("Dataset split into train and validation.")

    # Optionally, use a larger subset if H100 memory allows
    max_train_samples = min(20000, len(dataset['train']))  # Adjust as needed
    max_val_samples = min(2000, len(dataset['validation']))  # Adjust as needed
    dataset['train'] = dataset['train'].select(range(max_train_samples))
    dataset['validation'] = dataset['validation'].select(range(max_val_samples))
    logger.info(f"Dataset subset selected: {max_train_samples} train samples, {max_val_samples} validation samples.")

    log_memory_usage("after dataset loading and selection")

    # ============================================
    # 5. Load Model and Tokenizer
    # ============================================
    model_name = "meta-llama/Llama-3.2-1B"  # Replace with your model's path if different
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    logger.info(f"Model and tokenizer loaded and moved to {device} with dtype {torch_dtype}.")

    log_memory_usage("after model loading")

    # ============================================
    # 6. Preprocessing Function
    # ============================================
    def preprocess_function(examples):
        inputs = [
            f"{instruction} {input_text}"
            for instruction, input_text in zip(examples["instruction"], examples["input"])
        ]
        targets = examples["output"]
        model_inputs = tokenizer(
            inputs, max_length=256, padding="max_length", truncation=True
        )
        model_targets = tokenizer(
            targets, max_length=256, padding="max_length", truncation=True
        )
        labels = [
            [
                (token if token != tokenizer.pad_token_id else -100)
                for token in label
            ]
            for label in model_targets["input_ids"]
        ]
        model_inputs["labels"] = labels
        return model_inputs

    # ============================================
    # 7. Apply Preprocessing
    # ============================================
    log_memory_usage("before preprocessing")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    log_memory_usage("after preprocessing")
    logger.info("Dataset preprocessing completed.")

    # ============================================
    # 8. Initialize Data Collator
    # ============================================
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ============================================
    # 9. Training Arguments Setup
    # ============================================
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        per_device_train_batch_size=100,          
        gradient_accumulation_steps=1,          
        fp16=True,                               
        bf16=True,                               
        num_train_epochs=5,                      
        logging_dir="./logs",
        logging_steps=50,                        
        save_steps=1000,                         
        save_total_limit=5,                      
        evaluation_strategy="steps",
        eval_steps=500,                          
        warmup_steps=1000,                       
        learning_rate=3e-5,                      
        lr_scheduler_type="linear",
        weight_decay=0.01,
        report_to=["tensorboard"],
        dataloader_pin_memory=True,              
        dataloader_num_workers=8,                
        remove_unused_columns=True,              
    )
    logger.info("Training arguments set with optimizations for H100.")

    log_memory_usage("after training arguments setup")

    # ============================================
    # 10. Create DataLoaders
    # ============================================
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory
    )

    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=training_args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory
    )
    logger.info("DataLoaders created.")

    log_memory_usage("after DataLoader creation")


    total_training_steps = (len(train_dataloader) // training_args.gradient_accumulation_steps) * training_args.num_train_epochs
    logger.info(f"Total training steps: {total_training_steps}")




    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=total_training_steps
    )
    logger.info("Optimizer and scheduler initialized.")

    log_memory_usage("after optimizer and scheduler setup")

    # ============================================
    # 13. Initialize AMP Scaler for Mixed Precision
    # ============================================
    scaler = torch.cuda.amp.GradScaler()

    # ============================================
    # 14. Evaluation Function
    # ============================================
    def evaluate(model, dataloader, device, step):
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                eval_loss += outputs.loss.item()
        eval_loss /= len(dataloader)
        logger.info(f"Evaluation loss after step {step}: {eval_loss:.4f}")

        # Free up memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log_memory_usage("after evaluation")

    # ============================================
    # 15. Checkpoint Saving Function
    # ============================================
    def save_checkpoint(model, tokenizer, epoch):
        checkpoint_dir = f"./results/checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # ============================================
    # 16. Final Model Saving Function
    # ============================================
    def save_final_model(model, tokenizer):
        final_model_dir = "./final_model"
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info(f"Training completed. Final model saved to {final_model_dir}")

    # ============================================
    # 17. Training Loop
    # ============================================
    def train():
        model.train()
        global_step = 0
        for epoch in range(int(training_args.num_train_epochs)):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{training_args.num_train_epochs}")
            for step, batch in enumerate(epoch_iterator):
                log_memory_usage("before batch processing")
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / training_args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                # Log loss
                if training_args.logging_steps > 0 and global_step % training_args.logging_steps == 0:
                    logger.info(
                        f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item() * training_args.gradient_accumulation_steps:.4f}"
                    )
                    log_memory_usage("after logging")

                # Evaluation
                if training_args.eval_steps > 0 and global_step % training_args.eval_steps == 0:
                    evaluate(model, eval_dataloader, device, global_step)
                    model.train()

                # Periodic Memory Cleanup
                if step % 1000 == 0 and step != 0:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    log_memory_usage("periodic memory cleanup")

            # Save checkpoint after each epoch
            save_checkpoint(model, tokenizer, epoch)

        # Save the final model after training
        save_final_model(model, tokenizer)

    # ============================================
    # 18. Run Training
    # ============================================
    train()

    # ============================================
    # 19. Final Evaluation
    # ============================================
    evaluate(model, eval_dataloader, device, "final")

if __name__ == "__main__":
    main()
