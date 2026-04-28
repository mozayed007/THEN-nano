import argparse
import json
import random

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig, THENGPT

PAD = 0
BOS = 1
MEM = 2
SLOT = 3
QUERY = 4
ANSWER = 5
SEP = 6
FILLER_TOKENS = [7, 8, 9, 10]
VALUE_TOKENS = [16, 17, 18, 19]
ANSWER_POSITION = 3
VOCAB_SIZE = 32
CHUNK_TOKENS = 8


def build_model(model_kind, device):
    config = GPTConfig(
        sequence_len=CHUNK_TOKENS,
        vocab_size=VOCAB_SIZE,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="L",
    )
    if model_kind == "gpt":
        model = GPT(config)
    else:
        model = THENGPT(config)
    model.to(device)
    model.init_weights()
    model.cos = model.cos.to(COMPUTE_DTYPE)
    model.sin = model.sin.to(COMPUTE_DTYPE)
    return model


def make_episode(value_token, filler_offset, device):
    filler_tokens = [FILLER_TOKENS[(filler_offset + i) % len(FILLER_TOKENS)] for i in range(3)]
    chunk1_raw = torch.tensor([BOS, MEM, SLOT, value_token, filler_tokens[0], filler_tokens[1], filler_tokens[2], SEP, PAD], dtype=torch.long, device=device)
    chunk2_raw = torch.tensor([BOS, QUERY, SLOT, ANSWER, value_token, SEP, PAD, PAD, PAD], dtype=torch.long, device=device)
    return chunk1_raw[:-1], chunk1_raw[1:], chunk2_raw[:-1], chunk2_raw[1:]


def make_batch(batch_size, device):
    chunk1_inputs = []
    chunk1_targets = []
    chunk2_inputs = []
    chunk2_targets = []
    values = []
    for batch_idx in range(batch_size):
        value_token = random.choice(VALUE_TOKENS)
        c1_in, c1_tg, c2_in, c2_tg = make_episode(value_token, batch_idx, device)
        chunk1_inputs.append(c1_in)
        chunk1_targets.append(c1_tg)
        chunk2_inputs.append(c2_in)
        chunk2_targets.append(c2_tg)
        values.append(value_token)
    return (
        torch.stack(chunk1_inputs, dim=0),
        torch.stack(chunk1_targets, dim=0),
        torch.stack(chunk2_inputs, dim=0),
        torch.stack(chunk2_targets, dim=0),
        torch.tensor(values, dtype=torch.long, device=device),
    )


def train_gpt(model, steps, batch_size, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        c1_in, c1_tg, c2_in, c2_tg, _ = make_batch(batch_size, device)
        loss = model(c1_in, c1_tg) + model(c2_in, c2_tg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def train_then(model, steps, batch_size, lr, device, persist_state):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        c1_in, c1_tg, c2_in, c2_tg, _ = make_batch(batch_size, device)
        loss1, state = model(c1_in, c1_tg, state=None, return_state=True)
        if persist_state:
            loss2, _ = model(c2_in, c2_tg, state=state, return_state=True)
        else:
            loss2, _ = model(c2_in, c2_tg, state=None, return_state=True)
        loss = loss1 + loss2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def evaluate_gpt(model, episodes, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for episode_idx in range(episodes):
            value_token = VALUE_TOKENS[episode_idx % len(VALUE_TOKENS)]
            _, _, c2_in, _ = make_episode(value_token, episode_idx, device)
            logits = model(c2_in.unsqueeze(0))
            pred = logits[:, ANSWER_POSITION, :].argmax(dim=-1)
            correct += int(pred.item() == value_token)
    return correct / episodes


def evaluate_then(model, episodes, device, mode):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for episode_idx in range(episodes):
            value_token = VALUE_TOKENS[episode_idx % len(VALUE_TOKENS)]
            c1_in, _, c2_in, _ = make_episode(value_token, episode_idx, device)
            state = None
            if mode == "persistent":
                _, state = model(c1_in.unsqueeze(0), state=None, return_state=True)
            elif mode == "shuffled":
                wrong_value = VALUE_TOKENS[(episode_idx + 1) % len(VALUE_TOKENS)]
                wrong_c1_in, _, _, _ = make_episode(wrong_value, episode_idx + 1, device)
                _, state = model(wrong_c1_in.unsqueeze(0), state=None, return_state=True)
            logits, _ = model(c2_in.unsqueeze(0), state=state, return_state=True)
            pred = logits[:, ANSWER_POSITION, :].argmax(dim=-1)
            correct += int(pred.item() == value_token)
    return correct / episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    gpt = build_model("gpt", device)
    then_reset = build_model("then", device)
    then_persistent = build_model("then", device)

    train_gpt(gpt, args.steps, args.batch_size, args.lr, device)
    train_then(then_reset, args.steps, args.batch_size, args.lr, device, persist_state=False)
    train_then(then_persistent, args.steps, args.batch_size, args.lr, device, persist_state=True)

    results = {
        "gpt_baseline": evaluate_gpt(gpt, args.eval_episodes, device),
        "then_reset": evaluate_then(then_reset, args.eval_episodes, device, mode="reset"),
        "then_persistent": evaluate_then(then_persistent, args.eval_episodes, device, mode="persistent"),
        "then_shuffled_state": evaluate_then(then_persistent, args.eval_episodes, device, mode="shuffled"),
    }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
