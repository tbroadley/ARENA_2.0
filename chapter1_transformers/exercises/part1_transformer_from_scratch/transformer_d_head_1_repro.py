# %%

import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import webbrowser

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == '__main__'


if MAIN:
	reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
	reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
	tokens = reference_gpt2.to_tokens(reference_text).to(device)
	logits, cache = reference_gpt2.run_with_cache(tokens)

@dataclass
class Config:
	d_model: int = 768
	debug: bool = True
	layer_norm_eps: float = 1e-5
	d_vocab: int = 50257
	init_range: float = 0.02
	n_ctx: int = 1024
	d_head: int = 64
	d_mlp: int = 3072
	n_heads: int = 12
	n_layers: int = 12

def rand_float_test(cls, shape):
	cfg = Config(debug=True)
	layer = cls(cfg).to(device)
	random_input = t.randn(shape).to(device)
	print("Input shape:", random_input.shape)
	output = layer(random_input)
	if isinstance(output, tuple): output = output[0]
	print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
	cfg = Config(debug=True)
	layer = cls(cfg).to(device)
	random_input = t.randint(100, 1000, shape).to(device)
	print("Input shape:", random_input.shape)
	output = layer(random_input)
	if isinstance(output, tuple): output = output[0]
	print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
	cfg = Config(debug=True)
	layer = cls(cfg).to(device)
	layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
	print("Input shape:", input.shape)
	output = layer(input)
	if isinstance(output, tuple): output = output[0]
	print("Output shape:", output.shape)
	try: reference_output = gpt2_layer(input)
	except: reference_output = gpt2_layer(input, input, input)
	print("Reference output shape:", reference_output.shape, "\n")
	comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
	print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")

class LayerNorm(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.w = nn.Parameter(t.ones(cfg.d_model))
		self.b = nn.Parameter(t.zeros(cfg.d_model))

	def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
		residual_mean = residual.mean(dim=-1, keepdim=True)
		residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()

		residual = (residual - residual_mean) / residual_std
		return residual * self.w + self.b


if MAIN:
	rand_float_test(LayerNorm, [2, 4, 768])
	load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])


class Embed(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
		nn.init.normal_(self.W_E, std=self.cfg.init_range)

	def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
		return self.W_E[tokens]


if MAIN:
	rand_int_test(Embed, [2, 4])
	load_gpt2_test(Embed, reference_gpt2.embed, tokens)

class PosEmbed(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
		nn.init.normal_(self.W_pos, std=self.cfg.init_range)

	def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
		batch, seq_len = tokens.shape
		return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)


if MAIN:
	rand_int_test(PosEmbed, [2, 4])
	load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)


class Attention(nn.Module):
	IGNORE: Float[Tensor, ""]

	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
		self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
		self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
		self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
		self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
		self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
		self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
		self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
		nn.init.normal_(self.W_Q, std=self.cfg.init_range)
		nn.init.normal_(self.W_K, std=self.cfg.init_range)
		nn.init.normal_(self.W_V, std=self.cfg.init_range)
		nn.init.normal_(self.W_O, std=self.cfg.init_range)
		self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

	def forward(
		self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
	) -> Float[Tensor, "batch posn d_model"]:
		# Calculate query, key and value vectors
		q = einops.einsum(
			normalized_resid_pre, self.W_Q,
			"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
		) + self.b_Q
		k = einops.einsum(
			normalized_resid_pre, self.W_K,
			"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
		) + self.b_K
		v = einops.einsum(
			normalized_resid_pre, self.W_V,
			"batch posn d_model, nheads d_model d_head -> batch posn nheads d_head", 
		) + self.b_V

		# Calculate attention scores, then scale and mask, and apply softmax to get probabilities
		attn_scores = einops.einsum(
			q, k,
			"batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K", 
		)
		attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
		attn_pattern = attn_scores_masked.softmax(-1)

		# Take weighted sum of value vectors, according to attention probabilities
		z = einops.einsum(
			v, attn_pattern,
			"batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head", 
		)

		# Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
		attn_out = einops.einsum(
			z, self.W_O,
			"batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model", 
		) + self.b_O

		return attn_out

	def apply_causal_mask(
		self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
	) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
		'''
		Applies a causal mask to attention scores, and returns masked scores.
		'''
		# Define a mask that is True for all positions we want to set probabilities to zero for
		all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
		mask = t.triu(all_ones, diagonal=1).bool()
		# Apply the mask to attention scores, then return the masked scores
		attn_scores.masked_fill_(mask, self.IGNORE)
		return attn_scores



if MAIN:
	rand_float_test(Attention, [2, 4, 768])
	load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])

class MLP(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
		self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
		self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
		self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
		nn.init.normal_(self.W_in, std=self.cfg.init_range)
		nn.init.normal_(self.W_out, std=self.cfg.init_range)

	def forward(
		self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
	) -> Float[Tensor, "batch posn d_model"]:
		pre = einops.einsum(
			normalized_resid_mid, self.W_in,
			"batch position d_model, d_model d_mlp -> batch position d_mlp", 
		) + self.b_in
		post = gelu_new(pre)
		mlp_out = einops.einsum(
			post, self.W_out,
			"batch position d_mlp, d_mlp d_model -> batch position d_model", 
		) + self.b_out
		return mlp_out



if MAIN:
	rand_float_test(MLP, [2, 4, 768])
	load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])


class TransformerBlock(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.ln1 = LayerNorm(cfg)
		self.attn = Attention(cfg)
		self.ln2 = LayerNorm(cfg)
		self.mlp = MLP(cfg)

	def forward(
		self, resid_pre: Float[Tensor, "batch position d_model"]
	) -> Float[Tensor, "batch position d_model"]:
		resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
		resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
		return resid_post
		
		

if MAIN:
	rand_float_test(TransformerBlock, [2, 4, 768])
	load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])


class Unembed(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
		nn.init.normal_(self.W_U, std=self.cfg.init_range)
		self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

	def forward(
		self, normalized_resid_final: Float[Tensor, "batch position d_model"]
	) -> Float[Tensor, "batch position d_vocab"]:
		return einops.einsum(
			normalized_resid_final, self.W_U,
			"batch posn d_model, d_model d_vocab -> batch posn d_vocab",
		) + self.b_U
		# Or, could just do `normalized_resid_final @ self.W_U + self.b_U`



if MAIN:
	rand_float_test(Unembed, [2, 4, 768])
	load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])


class DemoTransformer(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.embed = Embed(cfg)
		self.pos_embed = PosEmbed(cfg)
		self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
		self.ln_final = LayerNorm(cfg)
		self.unembed = Unembed(cfg)

	def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
		residual = self.embed(tokens) + self.pos_embed(tokens)
		for block in self.blocks:
			residual = block(residual)
		logits = self.unembed(self.ln_final(residual))
		return logits



if MAIN:
	rand_int_test(DemoTransformer, [2, 4])
	load_gpt2_test(DemoTransformer, reference_gpt2, tokens)



def get_log_probs(
	logits: Float[Tensor, "batch posn d_vocab"], 
	tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
	
	log_probs = logits.log_softmax(dim=-1)
	# Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
	log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

	return log_probs_for_tokens

@dataclass
class TransformerTrainingArgs():
	batch_size = 8
	max_epochs = 1
	max_steps = 1000
	log_every = 10
	lr = 1e-3
	weight_decay = 1e-2
	log_dir: str = os.getcwd() + "/logs"
	log_name: str = "day1-transformer"
	run_name: Optional[str] = None
	log_every_n_steps: int = 1


if MAIN:
	args = TransformerTrainingArgs()


if MAIN:
	dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
	print(dataset)
	print(dataset[0]['text'][:100])


class LitTransformer(pl.LightningModule):
	def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer, data_loader: DataLoader):
		super().__init__()
		self.model = model
		self.cfg = model.cfg
		self.args = args
		self.data_loader = data_loader

	def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
		logits = self.model(tokens)
		return logits

	def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Float[Tensor, ""]:
		'''
		Here you compute and return the training loss and some additional metrics for e.g. 
		the progress bar or logger.
		'''
		tokens = batch["tokens"].to(device)
		logits = self.model(tokens)
		loss = -get_log_probs(logits, tokens).mean()
		self.log("train_loss", loss)
		return loss

	def configure_optimizers(self):
		'''
		Choose what optimizers and learning-rate schedulers to use in your optimization.
		'''
		optimizer = t.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
		return optimizer
	
	def train_dataloader(self):
		return self.data_loader


if MAIN:
    for d_head in [1, 64]:
        model_cfg = Config(
            debug=False, 
            d_model=256, 
            n_heads=256//d_head, 
            d_head=d_head,
            d_mlp=1024, 
            n_layers=2, 
            n_ctx=256, 
            d_vocab=reference_gpt2.cfg.d_vocab
        )
        model = DemoTransformer(model_cfg)

        tokenized_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model.cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
        data_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        litmodel = LitTransformer(args, model, data_loader)
        logger = WandbLogger(save_dir=args.log_dir, project=args.log_name, name=args.run_name)
        
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=logger,
            log_every_n_steps=args.log_every_n_steps
        )
        trainer.fit(model=litmodel, train_dataloaders=litmodel.data_loader)
        wandb.finish()