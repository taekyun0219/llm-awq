import torch
from datasets import load_dataset


def _conversation_to_text(conversations, tokenizer):
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass

    parts = []
    for turn in conversations:
        role = str(turn.get("role", "user")).strip()
        content = str(turn.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _collect_samples(dataset, tokenizer, n_samples, block_size, mode):
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0

    for row in dataset:
        if mode == "manta":
            line = _conversation_to_text(row["conversations"], tokenizer)
        else:
            line = row["text"]

        line = line.strip()
        line_encoded = tokenizer.encode(line, add_special_tokens=False)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    return samples


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        samples = _collect_samples(dataset, tokenizer, n_samples, block_size, mode="text")
    elif data == "manta":
        dataset = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
        samples = _collect_samples(dataset, tokenizer, n_samples, block_size, mode="manta")
    elif data == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train")
        samples = _collect_samples(dataset, tokenizer, n_samples, block_size, mode="text")
    elif data == "manta_c4":
        n_manta = n_samples // 2
        n_c4 = n_samples - n_manta
        manta_ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
        c4_ds = load_dataset("allenai/c4", "en", split="train")
        samples = _collect_samples(manta_ds, tokenizer, n_manta, block_size, mode="manta")
        samples += _collect_samples(c4_ds, tokenizer, n_c4, block_size, mode="text")
    else:
        raise NotImplementedError
    if len(samples) == 0:
        raise RuntimeError(
            f"No calibration samples collected for data={data}. "
            "Try increasing n_samples or decreasing block_size."
        )
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    if n_split == 0:
        raise RuntimeError(
            f"Insufficient calibration tokens for block_size={block_size}. "
            "Try increasing n_samples or reducing block_size."
        )
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
