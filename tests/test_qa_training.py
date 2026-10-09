import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qa_training", ROOT / "code" / "qa_training.py")
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


class DataPreparationTests(unittest.TestCase):
    def write_csv(self, directory, version, headers, rows):
        with (Path(directory) / f"{version}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)

    def test_v1_uses_human_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_csv(directory, "v1", ["soru", "insan cevabı", "makine cevabı"], [["Question?", "Human", "Machine"]])
            self.assertEqual(qa.load_qa_data(directory, "v1"), [{"soru": "Question?", "çıktı": "Human"}])

    def test_v2_accepts_spaced_headers_and_optional_input(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_csv(directory, "v2", ["", "talimat", " giriş", " çıktı"], [[0, "Explain", " context ", "Answer"], [1, "Question?", "", "Other"]])
            self.assertEqual(qa.load_qa_data(directory, "v2"), [
                {"soru": "Explain context", "çıktı": "Answer"},
                {"soru": "Question?", "çıktı": "Other"},
            ])

    def test_rejects_incomplete_data(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_csv(directory, "v3", ["soru", "çıktı"], [["Question?", ""]])
            with self.assertRaisesRegex(ValueError, "row 2"):
                qa.load_qa_data(directory, "v3")
            self.write_csv(directory, "test", ["soru"], [["Question?"]])
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                qa.load_qa_data(directory, "test")

    def test_rejects_unknown_version_and_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                qa.load_qa_data(directory, "../invalid")
            self.write_csv(directory, "v3", ["soru", "çıktı"], [])
            with self.assertRaisesRegex(ValueError, "no question-answer pairs"):
                qa.load_qa_data(directory, "v3")

    def test_checked_in_csvs_match_expected_schemas(self):
        for version in ("v1", "v2", "v3", "test"):
            with self.subTest(version=version):
                examples = qa.load_qa_data(ROOT / "Data", version)
                self.assertTrue(examples)
                self.assertEqual(set(examples[0]), {"soru", "çıktı"})
                self.assertFalse(examples[0]["soru"].startswith("Soru:"))


def fixture_tokenizer(reverse=False):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokens = ["<eos>", "<unk>", "Soru", ":", "Cevap", "Question", "?", "Answer", "Other", "Explain", "context"]
    if reverse:
        tokens[2:] = reversed(tokens[2:])
    tokenizer = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>", pad_token="<eos>", unk_token="<unk>")


class TokenizationTests(unittest.TestCase):
    def test_labels_align_with_full_sequence_and_preserve_eos(self):
        tokenizer = fixture_tokenizer()
        batch = qa.tokenize_qa_batch({"soru": ["Question?"], "çıktı": ["Answer"]}, tokenizer, 16)
        expected = tokenizer("Soru: Question? Cevap: Answer", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        self.assertEqual(batch["input_ids"][0][:len(expected)], expected)
        self.assertEqual(batch["labels"][0][:len(expected)], expected)
        self.assertEqual(batch["labels"][0][len(expected):], [-100] * (16 - len(expected)))
        self.assertEqual(batch["attention_mask"][0], [1] * len(expected) + [0] * (16 - len(expected)))

    def test_uses_selected_tokenizer_and_bounds_truncation(self):
        examples = {"soru": ["Question?", "Explain context"], "çıktı": ["Answer", "Other"]}
        first = qa.tokenize_qa_batch(examples, fixture_tokenizer(), 5)
        second = qa.tokenize_qa_batch(examples, fixture_tokenizer(reverse=True), 5)
        self.assertNotEqual(first["input_ids"], second["input_ids"])
        for row in second["input_ids"]:
            self.assertEqual(len(row), 5)
            self.assertEqual(row[-1], fixture_tokenizer().eos_token_id)

    def test_rejects_incompatible_lengths_and_missing_special_tokens(self):
        tokenizer = fixture_tokenizer()
        with self.assertRaises(ValueError):
            qa.tokenize_qa_batch({"soru": ["Question?"], "çıktı": []}, tokenizer)
        with self.assertRaises(ValueError):
            qa.tokenize_qa_batch({"soru": ["Question?"], "çıktı": ["Answer"]}, tokenizer, 1)
        tokenizer.pad_token = None
        with self.assertRaisesRegex(ValueError, "padding"):
            qa.tokenize_qa_batch({"soru": ["Question?"], "çıktı": ["Answer"]}, tokenizer)


class GenerationBudgetTests(unittest.TestCase):
    def test_rejects_budget_smaller_than_answer_delimiter(self):
        tokenizer = fixture_tokenizer()
        model = Mock()
        model.config.max_position_embeddings = 32
        with self.assertRaisesRegex(ValueError, "complete answer delimiter"):
            qa.generate_answer(model, tokenizer, "Question?", 31)
        model.generate.assert_not_called()

    def test_smallest_valid_budget_retains_complete_answer_delimiter(self):
        import torch

        tokenizer = fixture_tokenizer()
        model = Mock()
        model.config.max_position_embeddings = 32
        model.device = torch.device("cpu")
        model.generate.side_effect = lambda **kwargs: kwargs["input_ids"]
        self.assertEqual(qa.generate_answer(model, tokenizer, "Question? " * 30, 30), "")
        model.generate.assert_called_once()
        input_ids = model.generate.call_args.kwargs["input_ids"]
        self.assertEqual(input_ids.shape[1], 2)
        self.assertEqual(input_ids[0].tolist(), tokenizer(" Cevap:", add_special_tokens=False)["input_ids"])


class NotebookRuntimeTests(unittest.TestCase):
    def test_cells_compile_and_training_is_opt_in(self):
        notebook = json.loads((ROOT / "code" / "code.ipynb").read_text(encoding="utf-8"))
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for index, source in enumerate(sources):
            compile(source, f"notebook-cell-{index}", "exec")
        model_cell = next(source for source in sources if "AutoModelForCausalLM.from_pretrained" in source)
        with self.assertRaisesRegex(RuntimeError, "RUN_TRAINING"):
            exec(model_cell, {"RUN_TRAINING": False})

    def test_tiny_model_trains_saves_reloads_and_generates_offline(self):
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, TrainingArguments, set_seed

        torch.set_num_threads(1)
        set_seed(42)
        tokenizer = fixture_tokenizer()
        examples = [{"soru": "Question?", "çıktı": "Answer"}, {"soru": "Explain context", "çıktı": "Other"}]
        notebook = json.loads((ROOT / "code" / "code.ipynb").read_text(encoding="utf-8"))
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Data").mkdir()
            (root / "Data" / "fixture.csv").write_text("soru,answer\nQuestion?,Answer\n", encoding="utf-8")
            base_path = root / "base"
            base = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=32, n_ctx=32, n_embd=16, n_layer=1, n_head=2, bos_token_id=0, eos_token_id=0, pad_token_id=0))
            base.save_pretrained(base_path)
            tokenizer.save_pretrained(base_path)
            state = {
                "ROOT": root, "RUN_NAME": "fixture", "DATASET_VERSION": "fixture",
                "SEED": 42, "MAX_LENGTH": 16, "train_examples": examples,
                "MODELS": {"fixture": str(base_path)}, "MODEL_SIZE": "fixture",
                "MODEL_REVISION": "0" * 40, "RUN_TRAINING": True,
                "tokenize_qa_batch": qa.tokenize_qa_batch,
            }
            exec(next(source for source in sources if "AutoModelForCausalLM.from_pretrained" in source), state)
            self.assertEqual(state["resolved_revision"], "0" * 40)
            peft_model, tokenizer = state["peft_model"], state["tokenizer"]
            state["TrainingArguments"] = lambda **kwargs: TrainingArguments(**{**kwargs, "use_cpu": True, "fp16": False, "disable_tqdm": True})
            exec(next(source for source in sources if "train_dataset = Dataset.from_list" in source), state)
            train_source = next(source for source in sources if "training_args = TrainingArguments" in source)
            configuration, training = train_source.split("trainer = Trainer", 1)
            exec(configuration, state)
            before = {name: value.detach().clone() for name, value in peft_model.named_parameters() if value.requires_grad}
            exec("trainer = Trainer" + training, state)
            self.assertEqual(state["trainer"].state.global_step, 1)
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in peft_model.named_parameters() if name in before))
            exec(next(source for source in sources if "run_metadata =" in source), state)
            adapter_path = root / "saved_models" / "fixture"
            self.assertTrue((adapter_path / "adapter_model.safetensors").is_file())
            restored = AutoPeftModelForCausalLM.from_pretrained(adapter_path, local_files_only=True, use_safetensors=True)
            restored_tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
            peft_model.eval()
            restored.eval()
            input_ids = torch.tensor([[2, 3, 5]])
            with torch.inference_mode():
                torch.testing.assert_close(peft_model(input_ids).logits, restored(input_ids).logits)
            self.assertEqual(qa.generate_answer(peft_model, tokenizer, "Question?", 4), qa.generate_answer(restored, restored_tokenizer, "Question?", 4))
            with patch.object(restored, "generate", wraps=restored.generate) as generate:
                self.assertIsInstance(qa.generate_answer(restored, restored_tokenizer, "Question? " * 30, 4), str)
                input_ids = generate.call_args.kwargs["input_ids"]
                self.assertEqual(input_ids.shape[1], 28)
                self.assertEqual(input_ids[0, -2:].tolist(), restored_tokenizer("Cevap:", add_special_tokens=False)["input_ids"])
            with self.assertRaises(ValueError):
                qa.generate_answer(restored, restored_tokenizer, "Question?", 32)


if __name__ == "__main__":
    unittest.main()
