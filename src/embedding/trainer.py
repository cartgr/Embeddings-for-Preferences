"""Training pipeline for preference-aware embeddings."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from .dataset import InBatchNegativesDataset, TripletDataset
from .model import get_device, load_model, save_model

# Try to import peft for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvaluationCallback(TrainerCallback):
    """Callback to evaluate on GSC datasets and triplets during training."""

    def __init__(
        self,
        model: SentenceTransformer,
        train_triplets: List = None,
        eval_steps: int = 100,
        gsc_data_dir: str = "data/external/gsc",
        triplet_sample_size: int = 500,
    ):
        """Initialize callback.

        Args:
            model: The model being trained
            train_triplets: List of training triplets for accuracy eval
            eval_steps: Evaluate every N steps
            gsc_data_dir: Path to GSC data directory
            triplet_sample_size: Number of triplets to sample for eval
        """
        self.model = model
        self.train_triplets = train_triplets
        self.eval_steps = eval_steps
        self.gsc_data_dir = Path(gsc_data_dir)
        self.triplet_sample_size = triplet_sample_size
        self.datasets = None
        self.evaluator = None
        self._initialized = False

    def _lazy_init(self):
        """Lazily initialize GSC datasets and evaluator."""
        if self._initialized:
            return

        try:
            from ..evaluation.gsc_loader import GSCLoader
            from ..evaluation.gsc_evaluator import GSCEvaluator

            loader = GSCLoader(str(self.gsc_data_dir))
            self.datasets = []

            # Load chatbot dataset
            try:
                chatbot = loader.load_chatbot_personalization()
                self.datasets.append(chatbot)
                logger.info(f"Loaded GSC chatbot dataset: {len(chatbot.validation_users)} users")
            except Exception as e:
                logger.warning(f"Could not load chatbot dataset: {e}")

            # Load abortion dataset
            try:
                abortion = loader.load_abortion()
                self.datasets.append(abortion)
                logger.info(f"Loaded GSC abortion dataset: {len(abortion.validation_users)} users")
            except Exception as e:
                logger.warning(f"Could not load abortion dataset: {e}")

            if self.datasets:
                self.evaluator = GSCEvaluator(self.model)
                logger.info(f"GSC evaluation enabled on {len(self.datasets)} datasets")
            else:
                logger.warning("No GSC datasets found, GSC evaluation disabled")

        except Exception as e:
            logger.warning(f"Failed to load GSC datasets: {e}")
            self.datasets = []

        self._initialized = True

    def _eval_triplet_accuracy(self) -> float:
        """Evaluate triplet accuracy on a sample of training triplets."""
        if not self.train_triplets:
            return 0.0

        import random
        sample = random.sample(
            self.train_triplets,
            min(self.triplet_sample_size, len(self.train_triplets))
        )

        correct = 0
        for triplet in sample:
            embeddings = self.model.encode(
                [triplet.anchor_text, triplet.pos_text, triplet.neg_text],
                convert_to_tensor=True,
            )
            anchor, pos, neg = embeddings[0], embeddings[1], embeddings[2]

            pos_sim = F.cosine_similarity(anchor.unsqueeze(0), pos.unsqueeze(0))
            neg_sim = F.cosine_similarity(anchor.unsqueeze(0), neg.unsqueeze(0))

            if pos_sim > neg_sim:
                correct += 1

        return correct / len(sample)

    def on_step_end(self, args, state, control, **kwargs):
        """Evaluate periodically."""
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return

        self._lazy_init()

        logger.info(f"Running evaluation at step {state.global_step}...")

        try:
            import wandb

            # Triplet accuracy on training data
            if self.train_triplets:
                triplet_acc = self._eval_triplet_accuracy()
                logger.info(f"  Triplet accuracy: {triplet_acc:.3f}")

                if wandb.run is not None:
                    wandb.log({
                        "eval/triplet_accuracy": triplet_acc,
                    }, step=state.global_step)

            # GSC evaluation
            if self.datasets and self.evaluator:
                # Update evaluator's model reference
                self.evaluator.model = self.model

                for dataset in self.datasets:
                    result = self.evaluator.evaluate(dataset, model_name="training")

                    if wandb.run is not None:
                        wandb.log({
                            f"gsc/{dataset.name}/spearman": result.spearman_correlation,
                            f"gsc/{dataset.name}/pearson": result.pearson_correlation,
                            f"gsc/{dataset.name}/mae": result.mae,
                            f"gsc/{dataset.name}/pairwise_acc": result.pairwise_accuracy,
                        }, step=state.global_step)

                    logger.info(
                        f"  GSC {dataset.name}: MAE={result.mae:.3f}, "
                        f"Pairwise={result.pairwise_accuracy:.3f}, "
                        f"Spearman={result.spearman_correlation:.3f}"
                    )

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")


class BradleyTerryLoss(nn.Module):
    """Bradley-Terry loss for triplet training.

    L = -log(σ((sim(anchor, pos) - sim(anchor, neg)) / temperature))

    This is the probabilistic formulation that directly models
    P(pos > neg | anchor) without requiring a margin hyperparameter.
    Temperature < 1.0 sharpens the loss, encouraging larger margins.

    If proj_dim is set, a learned linear projection W: d -> proj_dim is applied
    before computing cosine similarity. The resulting embedding is the low-rank
    "preference embedding" e(x) = normalize(W f(x)).
    """

    def __init__(self, model: SentenceTransformer, temperature: float = 1.0,
                 proj_dim: int | None = None, input_dim: int = 768):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.proj_dim = proj_dim
        if proj_dim is not None:
            self.proj = nn.Linear(input_dim, proj_dim, bias=False)
            nn.init.orthogonal_(self.proj.weight)
        else:
            self.proj = None

    def forward(self, sentence_features, labels=None):
        # sentence_features is a list of dicts with input_ids, attention_mask, etc.
        # For triplets: [anchor_features, pos_features, neg_features]

        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]

        if self.proj is not None:
            embeddings = [self.proj(e) for e in embeddings]

        anchor_emb, pos_emb, neg_emb = embeddings

        # Cosine similarities
        pos_sim = F.cosine_similarity(anchor_emb, pos_emb)
        neg_sim = F.cosine_similarity(anchor_emb, neg_emb)

        # Bradley-Terry: -log(σ((pos_sim - neg_sim) / τ))
        loss = -F.logsigmoid((pos_sim - neg_sim) / self.temperature).mean()

        return loss


class DotProductBTLoss(nn.Module):
    """Bradley-Terry loss with unnormalized dot product scoring.

    Score: score(a,x) = u·v_x (raw dot product)
    Loss: -log(σ(score(a,p) - score(a,n)))
    Regularization: η * (||u||² + ||v_p||² + ||v_n||²)
    """

    def __init__(self, model: SentenceTransformer, reg_weight: float = 1e-4):
        super().__init__()
        self.model = model
        self.reg_weight = reg_weight

    def forward(self, sentence_features, labels=None):
        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        anchor_emb, pos_emb, neg_emb = embeddings

        # Raw dot products (no normalization)
        pos_score = torch.sum(anchor_emb * pos_emb, dim=1)
        neg_score = torch.sum(anchor_emb * neg_emb, dim=1)

        # BT loss
        bt_loss = -F.logsigmoid(pos_score - neg_score).mean()

        # Norm regularization to prevent blowup
        reg = self.reg_weight * (
            torch.mean(anchor_emb.pow(2)) +
            torch.mean(pos_emb.pow(2)) +
            torch.mean(neg_emb.pow(2))
        )

        return bt_loss + reg


class TheoryBTLoss(nn.Module):
    """Theory-motivated BT loss: score = τ*(u·v) - λ*||v||²

    This score function is derived from information-theoretic principles,
    where the norm correction term accounts for item "quality" or "popularity".

    Args:
        model: SentenceTransformer model
        tau_init: Initial value for temperature parameter τ
        lambda_init: Initial value for norm correction parameter λ
        reg_weight: Weight for L2 regularization on embeddings
    """

    def __init__(
        self,
        model: SentenceTransformer,
        tau_init: float = 1.0,
        lambda_init: float = 0.5,
        reg_weight: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.lam = nn.Parameter(torch.tensor(lambda_init))
        self.reg_weight = reg_weight

    def forward(self, sentence_features, labels=None):
        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        anchor_emb, pos_emb, neg_emb = embeddings

        # Dot products
        pos_dot = torch.sum(anchor_emb * pos_emb, dim=1)
        neg_dot = torch.sum(anchor_emb * neg_emb, dim=1)

        # Squared norms
        pos_norm_sq = torch.sum(pos_emb.pow(2), dim=1)
        neg_norm_sq = torch.sum(neg_emb.pow(2), dim=1)

        # Theory scores: τ*(u·v) - λ*||v||²
        pos_score = self.tau * pos_dot - self.lam * pos_norm_sq
        neg_score = self.tau * neg_dot - self.lam * neg_norm_sq

        # BT loss
        bt_loss = -F.logsigmoid(pos_score - neg_score).mean()

        # Norm regularization
        reg = self.reg_weight * (
            torch.mean(anchor_emb.pow(2)) +
            torch.mean(pos_emb.pow(2)) +
            torch.mean(neg_emb.pow(2))
        )

        return bt_loss + reg


class InfoNCELoss(nn.Module):
    """InfoNCE loss with hard negatives and in-batch negatives.

    For each anchor:
    - Positive: the pos_text from triplet
    - Hard negative: the neg_text from triplet
    - In-batch negatives: all other positives in the batch

    L = -log(exp(sim(a,p)/τ) / (exp(sim(a,p)/τ) + exp(sim(a,hard_neg)/τ) + Σ exp(sim(a,in_batch_neg)/τ)))
    """

    def __init__(self, model: SentenceTransformer, temperature: float = 0.05):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, sentence_features, labels=None):
        # sentence_features: [anchor_features, pos_features, neg_features]
        embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        anchor_emb, pos_emb, neg_emb = embeddings

        # Normalize embeddings
        anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)
        neg_emb = F.normalize(neg_emb, p=2, dim=1)

        batch_size = anchor_emb.size(0)

        # Positive scores: (batch_size,)
        pos_scores = torch.sum(anchor_emb * pos_emb, dim=1) / self.temperature

        # Hard negative scores: (batch_size,)
        hard_neg_scores = torch.sum(anchor_emb * neg_emb, dim=1) / self.temperature

        # In-batch negative scores: (batch_size, batch_size)
        # Each anchor compared to all positives in batch
        in_batch_scores = torch.matmul(anchor_emb, pos_emb.t()) / self.temperature

        # Mask out the diagonal (self-comparisons, which are the positives)
        mask = torch.eye(batch_size, device=anchor_emb.device, dtype=torch.bool)
        in_batch_scores = in_batch_scores.masked_fill(mask, float('-inf'))

        # Concatenate all negative scores: hard_neg + in_batch
        # Shape: (batch_size, 1 + batch_size - 1) = (batch_size, batch_size)
        all_neg_scores = torch.cat([
            hard_neg_scores.unsqueeze(1),  # (batch_size, 1)
            in_batch_scores,  # (batch_size, batch_size) with diagonal masked
        ], dim=1)

        # Compute log-softmax denominator
        # log(exp(pos) + sum(exp(negs)))
        all_scores = torch.cat([pos_scores.unsqueeze(1), all_neg_scores], dim=1)
        log_sum_exp = torch.logsumexp(all_scores, dim=1)

        # InfoNCE loss: -log(exp(pos) / sum(exp(all))) = -pos + log_sum_exp
        loss = (-pos_scores + log_sum_exp).mean()

        return loss


@dataclass
class TrainingConfig:
    """Configuration for embedding model training."""

    base_model: str = "all-MiniLM-L6-v2"
    output_dir: str = "data/models/embedding/finetuned"
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    train_ratio: float = 0.9
    seed: int = 42
    use_amp: bool = True  # Automatic mixed precision
    use_wandb: bool = True  # Enable wandb logging
    wandb_project: str = "preference-embeddings"
    logging_steps: int = 50  # Log every N steps
    eval_steps: int = 500  # Evaluate every N steps
    max_steps: int = -1  # Max training steps (-1 = use epochs)
    save_steps: int = 500  # Save checkpoint every N steps
    save_total_limit: int = 5  # Max checkpoints to keep
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps

    # LoRA configuration
    use_lora: bool = False
    use_dora: bool = False  # Weight-Decomposed Low-Rank Adaptation
    lora_r: int = 8  # Rank of LoRA decomposition
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None  # Auto-detect if None

    # Loss configuration
    loss_type: str = "bradley_terry"  # "bradley_terry" or "infonce"
    bt_temperature: float = 1.0  # Temperature for BT loss (< 1.0 = sharper, encourages larger margins)
    infonce_temperature: float = 0.05  # Temperature for InfoNCE loss

    # Projection head (Preference Embedding method).
    # If set, a rank-r learnable linear projection W: d -> proj_dim is applied
    # before cosine similarity. Training yields a proj_dim-dimensional embedding
    # space whose native cosine captures preference.
    proj_dim: Optional[int] = None

    # Evaluation configuration
    skip_final_eval: bool = False  # Skip slow final triplet evaluation

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load config from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data.get("training", data))


class EmbeddingTrainer:
    """Trainer for fine-tuning sentence transformer models on preference triplets."""

    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

    def _apply_lora(self, model: SentenceTransformer) -> SentenceTransformer:
        """Apply LoRA adapters to the underlying transformer model.

        Args:
            model: SentenceTransformer model

        Returns:
            Model with LoRA adapters applied
        """
        # Get the underlying transformer model (first module in SentenceTransformer)
        # SentenceTransformer structure: [0] = Transformer, [1] = Pooling
        transformer = model[0].auto_model

        # Determine target modules (query/value projections by default)
        target_modules = self.config.lora_target_modules
        if target_modules is None:
            # Auto-detect based on model architecture
            target_modules = ["query", "value"]  # Default for BERT-style
            for name, _ in transformer.named_modules():
                if "q_proj" in name:
                    target_modules = ["q_proj", "v_proj"]
                    break
                elif "query_key_value" in name:
                    target_modules = ["query_key_value"]
                    break
                elif ".q" in name and ".SelfAttention" in name:
                    # T5-style attention (encoder and decoder)
                    target_modules = ["q", "v"]
                    break
                elif "attention.attn.q" in name:
                    # MPNet-style attention
                    target_modules = ["q", "v"]
                    break

        logger.info(f"LoRA target modules: {target_modules}")
        logger.info(f"LoRA rank: {self.config.lora_r}, alpha: {self.config.lora_alpha}")

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            use_dora=self.config.use_dora,
        )

        # Apply LoRA to the transformer
        peft_model = get_peft_model(transformer, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        logger.info(
            f"LoRA trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        # Replace the transformer in the SentenceTransformer
        model[0].auto_model = peft_model

        return model

    def train(
        self,
        triplet_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        in_batch_negatives_baseline: bool = False,
    ) -> SentenceTransformer:
        """Train the embedding model.

        Args:
            triplet_path: Path to triplets JSONL file
            checkpoint_path: Optional path to resume from checkpoint
            in_batch_negatives_baseline: If True, use classic contrastive with in-batch negatives

        Returns:
            Trained SentenceTransformer model
        """
        # Load model
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            model = load_model(checkpoint_path, device=self.device)
        else:
            logger.info(f"Loading base model: {self.config.base_model}")
            model = load_model(self.config.base_model, device=self.device)

        # Apply LoRA if configured
        if self.config.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("peft library required for LoRA. Install with: pip install peft")

            logger.info("Applying LoRA adapters...")
            model = self._apply_lora(model)

        # Load and split dataset
        logger.info(f"Loading triplets from: {triplet_path}")

        if in_batch_negatives_baseline:
            logger.info("Using IN-BATCH NEGATIVES baseline (classic contrastive)")
            train_dataset = InBatchNegativesDataset.from_jsonl(triplet_path)
            # For validation, use standard triplets to measure actual triplet accuracy
            val_dataset = TripletDataset.from_jsonl(triplet_path)
            _, val_dataset = val_dataset.split(
                train_ratio=self.config.train_ratio, seed=self.config.seed
            )
        else:
            dataset = TripletDataset.from_jsonl(triplet_path)
            train_dataset, val_dataset = dataset.split(
                train_ratio=self.config.train_ratio,
                seed=self.config.seed,
            )

        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Setup loss function
        if in_batch_negatives_baseline:
            # MultipleNegativesRankingLoss: classic contrastive with in-batch negatives
            train_loss = losses.MultipleNegativesRankingLoss(model)
        elif self.config.loss_type == "infonce":
            # InfoNCE with hard negatives + in-batch negatives
            train_loss = InfoNCELoss(model, temperature=self.config.infonce_temperature)
            logger.info(f"Using InfoNCE loss (temperature={self.config.infonce_temperature})")
        else:
            # Bradley-Terry loss: -log(σ((sim(a,p) - sim(a,n)) / τ))
            # Probabilistic formulation without margin hyperparameter
            train_loss = BradleyTerryLoss(
                model,
                temperature=self.config.bt_temperature,
                proj_dim=self.config.proj_dim,
                input_dim=model.get_sentence_embedding_dimension(),
            )
            if self.config.proj_dim is not None:
                logger.info(f"Using rank-{self.config.proj_dim} projection head")
            logger.info(f"Using Bradley-Terry loss (temperature={self.config.bt_temperature})")

        # Calculate training steps
        steps_per_epoch = len(train_dataset) // self.config.batch_size
        if self.config.max_steps > 0:
            total_steps = self.config.max_steps
        else:
            total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        # Setup output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run name for wandb
        if in_batch_negatives_baseline:
            run_name = "in-batch-baseline"
        elif self.config.use_lora:
            run_name = f"lora-r{self.config.lora_r}-bradley-terry"
        else:
            run_name = "bradley-terry"

        # Training arguments with wandb integration
        training_args_kwargs = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": warmup_steps,
            "fp16": False,  # MPS doesn't support fp16
            "bf16": False,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "save_total_limit": getattr(self.config, 'save_total_limit', 5),
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "load_best_model_at_end": False,
            "report_to": "wandb" if self.config.use_wandb else "none",
            "run_name": run_name,
            "seed": self.config.seed,
        }

        # Use max_steps if specified, otherwise use epochs
        if self.config.max_steps > 0:
            training_args_kwargs["max_steps"] = self.config.max_steps
            logger.info(f"Training for {self.config.max_steps} steps")
        else:
            training_args_kwargs["num_train_epochs"] = self.config.epochs
            logger.info(f"Training for {self.config.epochs} epochs")

        args = SentenceTransformerTrainingArguments(**training_args_kwargs)

        # Initialize wandb manually to set project
        if self.config.use_wandb:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    "base_model": self.config.base_model,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "warmup_ratio": self.config.warmup_ratio,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "loss_type": "in_batch_negatives" if in_batch_negatives_baseline else "bradley_terry",
                    "device": self.device,
                },
                reinit=True,
            )

        # Convert to HuggingFace Dataset format
        hf_train_dataset = train_dataset.to_hf_dataset()

        # Setup evaluation callback (GSC + triplet accuracy)
        eval_callback = EvaluationCallback(
            model=model,
            train_triplets=train_dataset.triplets if hasattr(train_dataset, 'triplets') else None,
            eval_steps=self.config.eval_steps,
            gsc_data_dir="data/external/gsc",
            triplet_sample_size=500,
        )

        # Create trainer
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=hf_train_dataset,
            loss=train_loss,
            callbacks=[eval_callback],
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        logger.info(f"Saving model to: {output_dir}")
        model.save(str(output_dir))

        # If we used a projection head, save its weights alongside the model
        if self.config.proj_dim is not None and hasattr(train_loss, "proj") and train_loss.proj is not None:
            proj_path = Path(output_dir) / "projection.pt"
            torch.save({
                "weight": train_loss.proj.weight.detach().cpu(),
                "proj_dim": self.config.proj_dim,
                "input_dim": train_loss.proj.weight.shape[1],
            }, proj_path)
            logger.info(f"Saved projection head to {proj_path}")

        # Evaluate and log final metrics
        if not self.config.skip_final_eval:
            logger.info("Evaluating final model...")
            sample_size = min(1000, len(val_dataset.triplets))
            eval_triplets = val_dataset.triplets[:sample_size]
            eval_dataset = TripletDataset(eval_triplets)
            metrics = self.evaluate_model(model, eval_dataset)
            logger.info(f"Final triplet accuracy: {metrics['triplet_accuracy']:.3f}")

            if self.config.use_wandb:
                import wandb
                wandb.log({
                    "final/triplet_accuracy": metrics["triplet_accuracy"],
                    "final/correct": metrics["correct"],
                    "final/total": metrics["total"],
                })

        if self.config.use_wandb:
            import wandb
            wandb.finish()

        return model

    def evaluate_model(
        self,
        model: SentenceTransformer,
        val_dataset: TripletDataset,
    ) -> dict:
        """Evaluate model on validation set.

        Args:
            model: Trained model
            val_dataset: Validation dataset

        Returns:
            Dictionary of evaluation metrics
        """
        import torch

        correct = 0
        total = 0

        for triplet in val_dataset.triplets:
            embeddings = model.encode(
                [triplet.anchor_text, triplet.pos_text, triplet.neg_text],
                convert_to_tensor=True,
            )

            anchor_emb = embeddings[0]
            pos_emb = embeddings[1]
            neg_emb = embeddings[2]

            # Cosine similarity
            pos_sim = torch.nn.functional.cosine_similarity(
                anchor_emb.unsqueeze(0), pos_emb.unsqueeze(0)
            )
            neg_sim = torch.nn.functional.cosine_similarity(
                anchor_emb.unsqueeze(0), neg_emb.unsqueeze(0)
            )

            if pos_sim > neg_sim:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "triplet_accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
