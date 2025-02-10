# src/fine_tuning.py

import logging
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesSymmetricRankingLoss 
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer

logger = logging.getLogger(__name__)

def prepare_dataset(pairs):
    """
    Prepares a HuggingFace Dataset from a list of (textA, textB) pairs.
    Assumes all pairs are positive (score=1.0).
    If 'pairs' is already a Dataset, returns as is.
    """
    try:
        if isinstance(pairs, Dataset):
            return pairs

        data = {
            "sentence1": [pair[0] for pair in pairs],
            "sentence2": [pair[1] for pair in pairs],
            "score": [1.0] * len(pairs)  # All are positive
        }
        return Dataset.from_dict(data)

    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise

def fine_tune_sbert(
    model,
    train_pairs,
    num_epochs=1,
    batch_size=64,
    learning_rate=2e-5,
    output_dir="checkpoints/25_fine_tuned_model",
    warmup_ratio=0.1
):
    """
    Fine-tune a SentenceTransformer with symmetrical multiple negatives ranking loss.
    If your pairs (sentence1, sentence2) are truly interchangeable, 
    this can yield better alignment.
    """
    try:
        if not train_pairs:
            logger.warning("No training pairs found. Skipping fine-tuning.")
            return model

        # 1) Prepare the training & evaluation datasets
        logger.info("Preparing training and evaluation datasets...")
        train_dataset = prepare_dataset(train_pairs)
        # eval_dataset = train_dataset  # using the same for convenience/demo

        logger.info(f"Dataset prepared with {len(train_dataset)} training samples.")

        # 2) Configure training args
        logger.info("Configuring training arguments...")


        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            fp16=torch.cuda.is_available(),
            bf16=False,
            batch_sampler="no_duplicates",
            eval_strategy="no",
            eval_steps=100,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            logging_steps=100,
            run_name="sbert-fine-tuned-symmetric"
        )

        # 3)training loss: multiple negatives
        logger.info("Setting up MultipleNegativesSymmetricRankingLoss...")
        train_loss = MultipleNegativesSymmetricRankingLoss(model)

       
       
        # 5) Create a Trainer and train
        logger.info("Starting SBERT fine-tuning with symmetric MN loss...")
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss
        )
        trainer.train()


        # 7) Save the fine-tuned model
        logger.info(f"Saving fine-tuned model to {output_dir}...")
        model.save(output_dir)
        logger.info(f"Fine-tuned model saved successfully to {output_dir}")

        return model

    except Exception as e:
        logger.error(f"Error in fine_tune_sbert: {e}")
        raise
