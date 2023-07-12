# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP

t.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part3_indirect_object_identification").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, scatter, bar
import part3_indirect_object_identification.tests as tests

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

MAIN = __name__ == "__main__"

# %%

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

# %%

# Show column norms are the same (except first few, for fiddly bias reasons)
line([model.W_Q[0, 0].pow(2).sum(0), model.W_K[0, 0].pow(2).sum(0)])
# Show columns are orthogonal (except first few, again)
W_Q_dot_products = einops.einsum(
    model.W_Q[0, 0], model.W_Q[0, 0], "d_model d_head_1, d_model d_head_2 -> d_head_1 d_head_2"
)
imshow(W_Q_dot_products)

# %%

# Here is where we test on a single prompt
# Result: 70% probability on Mary, as we expect

example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
example_answer = " Mary"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# %%

prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
name_pairs = [
    (" John", " Mary"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]

# Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
prompts = [
    prompt.format(name) 
    for (prompt, names) in zip(prompt_format, name_pairs) for name in names[::-1] 
]
# Define the answers for each prompt, in the form (correct, incorrect)
answers = [names[::i] for names in name_pairs for i in (1, -1)]
# Define the answer tokens (same shape as the answers)
answer_tokens = t.concat([
    model.to_tokens(names, prepend_bos=False).T for names in answers
])

rprint(prompts)
rprint(answers)
rprint(answer_tokens)

table = Table("Prompt", "Correct", "Incorrect", title="Prompts & Answers:")

for prompt, answer in zip(prompts, answers):
    table.add_row(prompt, repr(answer[0]), repr(answer[1]))

rprint(table)

# %%

tokens = model.to_tokens(prompts, prepend_bos=True)
# Move the tokens to the GPU
tokens = tokens.to(device)
# Run the model and cache all activations
original_logits, cache = model.run_with_cache(tokens)

# %%

def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    batch, _, _ = logits.shape

    last_logits = logits[:, -1]
    correct_answer_tokens, incorrect_answer_tokens = answer_tokens.unbind(-1)

    batch_range = t.arange(batch)
    correct_answer_logits = last_logits[batch_range, correct_answer_tokens]
    incorrect_answer_logits = last_logits[batch_range, incorrect_answer_tokens]

    result = correct_answer_logits - incorrect_answer_logits
    return result if per_prompt else result.mean()


tests.test_logits_to_ave_logit_diff(logits_to_ave_logit_diff)

original_per_prompt_diff = logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True)
print("Per prompt logit difference:", original_per_prompt_diff)
original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
print("Average logit difference:", original_average_logit_diff)

cols = [
    "Prompt", 
    Column("Correct", style="rgb(0,200,0) bold"), 
    Column("Incorrect", style="rgb(255,0,0) bold"), 
    Column("Logit Difference", style="bold")
]
table = Table(*cols, title="Logit differences")

for prompt, answer, logit_diff in zip(prompts, answers, original_per_prompt_diff):
    table.add_row(prompt, repr(answer[0]), repr(answer[1]), f"{logit_diff.item():.3f}")

rprint(table)

# %%

# How could a Transformer implement this behaviour?
# Well, the explanation written down before was that it
# 1. detects names in the sentence
# 2. returns the name that isn't repeated
# How could it do that?
# Maybe we have an OV circuit that returns high-norm vectors for tokens that look like
# names or the start of names.
# Then in the next layer, these feed into an attention head through K-composition. So we end up
# with an attention head that pays a lot of attention to the name tokens.
# Then maybe if there are two name tokens, the two cancel out. Leaving only the name token that appears
# once in the sequence.
# Then that gets fed into another attention head that converts the "the name that appears once is X"
# info into "this token should be X".

# %%

answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)

correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions
print(f"Logit difference directions shape:", logit_diff_directions.shape)

# %%


# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type]. 

final_residual_stream: Float[Tensor, "batch seq d_model"] = cache["resid_post", -1]
print(f"Final residual stream shape: {final_residual_stream.shape}")
final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, -1, :]

# Apply LayerNorm scaling (to just the final sequence position)
# pos_slice is the subset of the positions we take - here the final token of each prompt
scaled_final_token_residual_stream = cache.apply_ln_to_stack(final_token_residual_stream, layer=-1, pos_slice=-1)

average_logit_diff = einops.einsum(
    scaled_final_token_residual_stream, logit_diff_directions,
    "batch d_model, batch d_model ->"
) / len(prompts)

print(f"Calculated average logit diff: {average_logit_diff:.10f}")
print(f"Original logit difference:     {original_average_logit_diff:.10f}")

t.testing.assert_close(average_logit_diff, original_average_logit_diff)

# %%

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given 
    stack of components in the residual stream.
    '''
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)

    average_logit_diffs = einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / residual_stack.shape[-2]

    return average_logit_diffs


t.testing.assert_close(
    residual_stack_to_logit_diff(final_token_residual_stream, cache),
    original_average_logit_diff
)

# %%

accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
# accumulated_residual has shape (component, batch, d_model)

logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(accumulated_residual, cache)

line(
    logit_lens_logit_diffs, 
    hovermode="x unified",
    title="Logit Difference From Accumulated Residual Stream",
    labels={"x": "Layer", "y": "Logit Diff"},
    xaxis_tickvals=labels,
    width=800
)

# %%

per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache)

line(
    per_layer_logit_diffs, 
    hovermode="x unified",
    title="Logit Difference From Each Layer",
    labels={"x": "Layer", "y": "Logit Diff"},
    xaxis_tickvals=labels,
    width=800
)

# %%

per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
per_head_residual = einops.rearrange(
    per_head_residual, 
    "(layer head) ... -> layer head ...", 
    layer=model.cfg.n_layers
)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache)

imshow(
    per_head_logit_diffs, 
    labels={"x":"Head", "y":"Layer"}, 
    title="Logit Difference From Each Head",
    width=600
)

# %%

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = t.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()



k = 3

for head_type in ["Positive", "Negative"]:

    # Get the heads with largest (or smallest) contribution to the logit difference
    top_heads = topk_of_Nd_tensor(per_head_logit_diffs * (1 if head_type=="Positive" else -1), k)

    # Get all their attention patterns
    attn_patterns_for_important_heads: Float[Tensor, "head q k"] = t.stack([
        cache["pattern", layer][:, head][1::2].mean(0)
        for layer, head in top_heads
    ])

    # Display results
    display(HTML(f"<h2>Top {k} {head_type} Logit Attribution Heads</h2>"))
    display(cv.attention.attention_heads(
        attention = attn_patterns_for_important_heads,
        tokens = model.to_str_tokens(tokens[1]),
        attention_head_names = [f"{layer}.{head}" for layer, head in top_heads],
    ))

# %%

# Very interesting. So all of these heads are mostly attending to the first token,
# except when we reach a place in the sequence where the next token could be either the
# subject or the indirect object. In that case, the heads start attending to the original
# appearances of the subject and indirect object in the prompt.

# What kind of info are they passing around? Something like "the next token is whatever I'm attending to"

# It's weird that " gave" is also attending to the first " Mary". Why?
# I think it's just because of the fact that we were visualizing the mean instead of a particular prompt.
# If I visualize the first prompt's logits, almost all attention is paid to the correct token
# after both " gave" and " to". And same if I just visualize the second prompt.

# It's also weird that the positive and negative logit attribution heads are attending
# to the same positions. Maybe the negative heads have a OV circuit that applies the opposite
# effect to the positive heads?

# %%

from transformer_lens import patching

# %%

clean_tokens = tokens
# Swap each adjacent pair to get corrupted tokens
indices = [i+1 if i % 2 == 0 else i-1 for i in range(len(tokens))]
corrupted_tokens = clean_tokens[indices]

print(
    "Clean string 0:    ", model.to_string(clean_tokens[0]), "\n"
    "Corrupted string 0:", model.to_string(corrupted_tokens[0])
)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

# %%

def ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"], 
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    corrupted_logit_diff: float = corrupted_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is 
    same as on corrupted input, and 1 when performance is same as on clean input.
    '''
    logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


t.testing.assert_close(ioi_metric(clean_logits).item(), 1.0)
t.testing.assert_close(ioi_metric(corrupted_logits).item(), 0.0)
t.testing.assert_close(ioi_metric((clean_logits + corrupted_logits) / 2).item(), 0.5)

# %%

act_patch_resid_pre = patching.get_act_patch_resid_pre(
    model = model,
    corrupted_tokens = corrupted_tokens,
    clean_cache = clean_cache,
    patching_metric = ioi_metric
)

labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]

imshow(
    act_patch_resid_pre, 
    labels={"x": "Position", "y": "Layer"},
    x=labels,
    title="resid_pre Activation Patching",
    width=600
)

# I didn't really get this graph at first.
# Let's explain it.
# So the x axis is the position in the sequence.
# And the y axis is the layer number.
# Each position in the graph is the linear metric of the logit diff (indirect object logit - 
# subject logit), given that we patched the residual stream at that particular layer and 
# for that particular position in the sequence.
# So this is saying, if we patch the residual stream in early layers, the only patching
# that matters is in the position of the second mention of the subject. This makes sense because
# this is the only token that differs between the clean and corrupted token sequences.
# But after layer 8, this position basically stops mattering. The only patching that affects the
# output anymore is in the last position in the sequence. So the model has moved the information
# about what should be the next token to the last position in the sequence by layer 9 or 10.

# %%

def patch_residual_component(
    corrupted_residual_component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    pos: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos d_model"]:
    '''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
    '''
    clean_resid_pre = clean_cache[hook.name]
    corrupted_residual_component[:, pos] = clean_resid_pre[:, pos]
    return corrupted_residual_component

def get_act_patch_resid_pre(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], float]
) -> Float[Tensor, "layer pos"]:
    '''
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    n_tokens = corrupted_tokens.shape[1]

    results = t.zeros(model.cfg.n_layers, n_tokens, device=device)

    for layer in tqdm(range(model.cfg.n_layers)):
        for pos in range(n_tokens):
            model.reset_hooks()

            logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks = [(
                utils.get_act_name("resid_pre", layer), 
                partial(patch_residual_component, pos=pos, clean_cache=clean_cache)
                )]
            )

            results[layer, pos] = patching_metric(logits)

    return results


act_patch_resid_pre_own = get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, ioi_metric)

t.testing.assert_close(act_patch_resid_pre, act_patch_resid_pre_own)

# %%

act_patch_block_every = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, ioi_metric)

imshow(
    act_patch_block_every,
    x=labels, 
    facet_col=0, # This argument tells plotly which dimension to split into separate plots
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"], # Subtitles of separate plots
    title="Logit Difference From Patched Attn Head Output", 
    labels={"x": "Sequence Position", "y": "Layer"},
    width=1000,
)

# %%

act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
    model, 
    corrupted_tokens, 
    clean_cache, 
    ioi_metric
)

imshow(
    act_patch_attn_head_out_all_pos, 
    labels={"y": "Layer", "x": "Head"}, 
    title="attn_head_out Activation Patching (All Pos)",
    width=600
)

# %%

def patch_head_vector(
    corrupted_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the clean cache.
    '''
    clean_head_vector = clean_cache[hook.name]
    corrupted_head_vector[:, :, head_index] = clean_head_vector[:, :, head_index]
    return corrupted_head_vector

def get_act_patch_attn_head_out_all_pos(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable
) -> Float[Tensor, "layer head"]:
    '''
    Returns an array of results of patching at all positions for each head in each
    layer, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = t.zeros(n_layers, n_heads, device=device)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(n_heads):
            model.reset_hooks()

            logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks = [(
                # Patching here is strange.
                # Neel Nanda says, "This is the linear combination of value vectors, i.e. 
                # it's the thing you multiply by WO​ before adding back into the residual stream. 
                # There's no point patching after the WO​ multiplication, because it will have 
                # the same effect, but take up more memory (since d_model is larger than d_head)."
                # But this is not borne out by the results. If I patch attn_out instead of z,
                # I get different results.
                # But I guess it should be equivalent. W_O should be the same across both runs of the 
                # model. So yeah, it shouldn't make so much of a difference. I guess multiplying by W_O
                # could be applying a weird, hard-to-understand linear transform or something?
                utils.get_act_name("z", layer),
                partial(patch_head_vector, head_index=head, clean_cache=clean_cache)
                )]
            )

            results[layer, head] = patching_metric(logits)

    return results


act_patch_attn_head_out_all_pos_own = get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, ioi_metric)

t.testing.assert_close(act_patch_attn_head_out_all_pos, act_patch_attn_head_out_all_pos_own)

imshow(
    act_patch_attn_head_out_all_pos_own,
    title="Logit Difference From Patched Attn Head Output", 
    labels={"x":"Head", "y":"Layer"},
    width=600
)

# %%

act_patch_attn_head_all_pos_every = patching.get_act_patch_attn_head_all_pos_every(
    model, 
    corrupted_tokens, 
    clean_cache, 
    ioi_metric
)

imshow(
    act_patch_attn_head_all_pos_every, 
    facet_col=0, 
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Activation Patching Per Head (All Pos)", 
    labels={"x": "Head", "y": "Layer"},
)

# %%


layer_7_8_value_heads = [[7, 3], [7, 9], [8, 6], [8, 10]]

attn_patterns_for_layer_7_8_value_heads: Float[Tensor, "head q k"] = t.stack([
    cache["pattern", layer][:, head][1::2].mean(0)
    for layer, head in layer_7_8_value_heads
])

display(cv.attention.attention_heads(
    attention = attn_patterns_for_layer_7_8_value_heads,
    tokens = model.to_str_tokens(tokens[1]),
    attention_head_names = [f"{layer}.{head}" for layer, head in layer_7_8_value_heads],
))

# These heads are paying attention to the second occurrence of the subject,
# most heavily on the tokens " gave" and " to". Places where we would expect not to see
# the subject, but instead the indirect object as the next token.
# So these heads are probably preventing the model from predicting the subject is the 
# next token in these places.

# %%

attn_pattern_for_layer_3_head_0 = cache["pattern", 3][:, 0][1::2].mean(0)

display(cv.attention.attention_heads(
    attention = t.stack([attn_pattern_for_layer_3_head_0]),
    tokens = model.to_str_tokens(tokens[1]),
    attention_head_names = ["3.0"],
))

# Mostly rests, except that the second occurrence of the subject attends to 
# the first occurrence.

# %%

# Overall explanation:
#   - 3.0, 5.5, 6.9 attend from the second occurrence of the subject to the first
#     occurrence of the subject. They move information about the fact that the subject
#     occurs twice into the position S2 (of the second subject).
#   - 7.3, 7.9, 8.6, 8.10 move this information from S2 to the last position. Along the way they
#     convert it into something like "the next token shouldn't be the subject".
#   - 9.6, 9.9, 10.0 attend strongly to the IO token because of that information saying that they shouldn't
#     attend to S1 (the first occurrence of the subject).

# %%

from part3_indirect_object_identification.ioi_dataset import NAMES, IOIDataset

# %%

N = 25
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=str(device)
)

print(ioi_dataset.sentences)

# %%

abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

# %%

def format_prompt(sentence: str) -> str:
    '''Format a prompt by underlining names (for rich print)'''
    return re.sub("(" + "|".join(NAMES) + ")", lambda x: f"[u bold dark_orange]{x.group(0)}[/]", sentence) + "\n"


def make_table(cols, colnames, title="", n_rows=5, decimals=4):
    '''Makes and displays a table, from cols rather than rows (using rich print)'''
    table = Table(*colnames, title=title)
    rows = list(zip(*cols))
    f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
    for row in rows[:n_rows]:
        table.add_row(*list(map(f, row)))
    rprint(table)


make_table(
    colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
    cols = [
        map(format_prompt, ioi_dataset.sentences), 
        model.to_string(ioi_dataset.s_tokenIDs).split(), 
        model.to_string(ioi_dataset.io_tokenIDs).split(), 
        map(format_prompt, abc_dataset.sentences), 
    ],
    title = "Sentences from IOI vs ABC distribution",
)

# %%

def logits_to_ave_logit_diff_2(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset = ioi_dataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()



model.reset_hooks(including_permanent=True)

ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)

ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()

print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")

make_table(
    colnames = ["IOI prompt", "IOI logit diff", "ABC prompt", "ABC logit diff"],
    cols = [
        map(format_prompt, ioi_dataset.sentences), 
        ioi_per_prompt_diff,
        map(format_prompt, abc_dataset.sentences), 
        abc_per_prompt_diff,
    ],
    title = "Sentences from IOI vs ABC distribution",
)

# %%

def ioi_metric_2(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float = ioi_average_logit_diff,
    corrupted_logit_diff: float = abc_average_logit_diff,
    ioi_dataset: IOIDataset = ioi_dataset,
) -> float:
    '''
    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset), 
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
    return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)


print(f"IOI metric (IOI dataset): {ioi_metric_2(ioi_logits_original):.4f}")
print(f"IOI metric (ABC dataset): {ioi_metric_2(abc_logits_original):.4f}")

# %%

def patch_sender_and_freeze_others(
    _: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index_to_freeze: Optional[int], 
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
) -> Float[Tensor, "batch pos head_index d_head"]:
    new_head_vector = new_cache[hook.name]
    result = orig_cache[hook.name].clone()
    if head_index_to_freeze is not None:
        result[:, :, head_index_to_freeze] = new_head_vector[:, :, head_index_to_freeze]
    return result

def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
) -> Float[Tensor, "layer head"]:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = t.zeros(n_layers, n_heads, device=device)

    for layer_to_freeze in tqdm(range(model.cfg.n_layers)):
        for head in range(n_heads):
            model.reset_hooks()

            logits = model.run_with_hooks(
                orig_dataset.toks,
                fwd_hooks = [(
                    utils.get_act_name("z", layer),
                    partial(
                        patch_sender_and_freeze_others,
                        head_index_to_freeze=head if layer == layer_to_freeze else None, 
                        new_cache=new_cache,
                        orig_cache=orig_cache,
                    )
                ) for layer in range(n_layers)]
            )

            results[layer_to_freeze, head] = patching_metric(logits)

    return results


path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(model, ioi_metric_2)

imshow(
    100 * path_patch_head_to_final_resid_post,
    title="Direct effect on logit difference",
    labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
    coloraxis=dict(colorbar_ticksuffix = "%"),
    width=600,
)

# %%

def patch_receiver(
    original: Tensor,
    hook: HookPoint,
    receiver_heads: List[Tuple[int, int]],
    patched_and_frozen_cache: ActivationCache,
) -> Tensor:
    new_head_vector = patched_and_frozen_cache[hook.name]
    for layer, head in receiver_heads:
        if layer != hook.layer():
            continue
        
        original[:, :, head] = new_head_vector[:, :, head]
    return original

def get_path_patch_head_to_heads(
    receiver_heads: List[Tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset: IOIDataset = abc_dataset,
    orig_dataset: IOIDataset = ioi_dataset,
    new_cache: Optional[ActivationCache] = abc_cache,
    orig_cache: Optional[ActivationCache] = ioi_cache,
) -> Float[Tensor, "layer head"]:
    '''
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = input to a later head (or set of heads)

    The receiver node is specified by receiver_heads and receiver_input.
    Example (for S-inhibition path patching the values):
        receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
        receiver_input = "v"

    Returns:
        tensor of metric values for every possible sender head
    '''
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = t.zeros(n_layers, n_heads, device=device)

    for layer_to_freeze in tqdm(range(model.cfg.n_layers)):
        for head in range(n_heads):
            model.reset_hooks()

            for layer in range(n_layers):
                model.add_hook(
                    utils.get_act_name("z", layer),
                    partial(
                        patch_sender_and_freeze_others,
                        head_index_to_freeze=head if layer == layer_to_freeze else None,
                        new_cache=new_cache,
                        orig_cache=orig_cache,
                    ),
                    level=1
                )

            _, patched_and_frozen_cache = model.run_with_cache(orig_dataset.toks)

            assert receiver_input in ("k", "q", "v")
            receiver_layers = set(next(zip(*receiver_heads)))
            receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
            receiver_hook_names_filter = lambda name: name in receiver_hook_names

            hook_fn = partial(
                patch_receiver,
                receiver_heads=receiver_heads,
                patched_and_frozen_cache=patched_and_frozen_cache,
            )
            logits = model.run_with_hooks(
                orig_dataset.toks,
                fwd_hooks = [(receiver_hook_names_filter, hook_fn)],
                return_type="logits"
            )


            results[layer_to_freeze, head] = patching_metric(logits)

    return results

model.reset_hooks()

s_inhibition_value_path_patching_results = get_path_patch_head_to_heads(
    receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
    receiver_input = "v",
    model = model,
    patching_metric = ioi_metric_2
)

imshow(
    100 * s_inhibition_value_path_patching_results,
    title="Direct effect on S-Inhibition Heads' values", 
    labels={"x": "Head", "y": "Layer", "color": "Logit diff.<br>variation"},
    width=600,
    coloraxis=dict(colorbar_ticksuffix = "%"),
)