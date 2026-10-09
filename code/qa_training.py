import csv
from pathlib import Path


def load_qa_data(data_dir, version):
    if version not in {"v1", "v2", "v3", "test"}:
        raise ValueError("Dataset must be v1, v2, v3, or test")
    path = Path(data_dir) / f"{version}.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        required = {"soru", "insan cevabı"} if version == "v1" else (
            {"talimat", "giriş", "çıktı"} if version == "v2" else {"soru", "çıktı"}
        )
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"{path.name} is missing required columns: {sorted(required)}")
        examples = []
        for row_number, row in enumerate(reader, start=2):
            if version == "v2":
                question = " ".join(
                    part.strip() for part in (row.get("talimat") or "", row.get("giriş") or "")
                    if part.strip()
                )
            else:
                question = (row.get("soru") or "").strip()
            answer_key = "insan cevabı" if version == "v1" else "çıktı"
            answer = (row.get(answer_key) or "").strip()
            if not question or not answer:
                raise ValueError(f"{path.name} row {row_number} has an empty question or answer")
            examples.append({"soru": question, "çıktı": answer})
    if not examples:
        raise ValueError(f"{path.name} has no question-answer pairs")
    return examples


def format_prompt(question):
    return f"Soru: {question.strip()} Cevap:"


def tokenize_qa_batch(examples, tokenizer, max_length=512):
    if max_length < 2:
        raise ValueError("max_length must leave room for text and an EOS token")
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define EOS and padding tokens")
    questions, answers = examples["soru"], examples["çıktı"]
    if len(questions) != len(answers):
        raise ValueError("Questions and answers must have equal lengths")
    texts = [f"{format_prompt(question)} {answer.strip()}" for question, answer in zip(questions, answers)]
    encoded = tokenizer(texts, truncation=True, max_length=max_length - 1, add_special_tokens=False)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for token_ids in encoded["input_ids"]:
        token_ids = list(token_ids) + [tokenizer.eos_token_id]
        padding_length = max_length - len(token_ids)
        batch["input_ids"].append(token_ids + [tokenizer.pad_token_id] * padding_length)
        batch["attention_mask"].append([1] * len(token_ids) + [0] * padding_length)
        batch["labels"].append(token_ids + [-100] * padding_length)
    return batch


def generate_answer(model, tokenizer, question, max_new_tokens=64):
    import torch

    context_length = model.config.max_position_embeddings
    if not 0 < max_new_tokens < context_length:
        raise ValueError("max_new_tokens must be positive and smaller than the model context")
    input_length = context_length - max_new_tokens
    delimiter_ids = tokenizer(" Cevap:", add_special_tokens=False)["input_ids"]
    if input_length < len(delimiter_ids):
        raise ValueError("max_new_tokens must leave room for the complete answer delimiter")
    encoded = tokenizer(format_prompt(question), return_tensors="pt", add_special_tokens=False)
    encoded = {key: value[:, -input_length:].to(model.device) for key, value in encoded.items()}
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
