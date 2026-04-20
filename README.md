# Tokenization of Numerical Expressions and Structured Data

**NLP 2026 · Case Study 1.9 · Innopolis University**
**Author:** Artem Panov

This repository contains the full experimental pipeline for Case Study 1.9: a four-notebook investigation of how tokenizer design — and specifically digit-level tokenization — affects what language models can do with numbers.

---

## TL;DR

Across four experiments we find that **tokenizer design has two almost-independent effects**:

- **At the embedding level (unsupervised, no gradient):** every tokenizer encodes in-distribution magnitude reasonably well, but **none extrapolates to unseen ranges** — not even the confirmed digit-level Qwen2.5. Regular inputs alone do not yield regular representations.
- **At the task level (with gradient):** digit-split preprocessing lifts out-of-distribution numerical reasoning by **+14 percentage points** on DistilBERT. Regularity pays off exactly when the model is trained on it.

The shift of modern LLMs to digit-level tokenization is therefore justified — but it is a **pre-training-time investment**, not a free architectural lunch.

Key findings:

1. **Tokenizers fragment the same quantity very differently.** Qwen2.5 (confirmed digit-level) uses 4.9 sub-tokens per integer vs. 1.4–2.5 for all others. This is by design and underpins Qwen's documented numerical-reasoning abilities.
2. **Pythia-160m's tokenizer is NOT digit-level in practice.** Despite being GPT-NeoX-family, it produces ~1.4 sub-tokens per small integer — similar to BLOOM. Family name does not imply digit-level behaviour; you need to measure.
3. **Magnitude IS encoded in-distribution** for every tokenizer, including Qwen2.5 (comparison accuracy > 0.90 for all 6 probed models).
4. **No tokenizer extrapolates** to integers an order of magnitude larger than training. R² OOD ranges from −15 to −45 for all 6 models — a language-model-wide problem, not a tokenizer-specific one.
5. **On Financial PhraseBank**, preprocessing of numbers has only a small effect on English encoders (ΔF1 ≤ 0.013). Context carries most of the sentiment signal. XLM-R underperforms (F1 = 0.48) because its multilingual tokenizer is a poor fit for dense financial English.
6. **On a controlled OOD numerical task**, digit-split preprocessing boosts DistilBERT OOD accuracy from 75.8% to 89.9% at ~72% longer sequences.

---

## Repository layout

```
tokenization-case-study/
├── notebooks/
│   ├── Notebook_01_Tokenizer_Fragmentation.ipynb       # CPU, ~5 min
│   ├── Notebook_02_Number_Representation_Probing.ipynb # T4 GPU, ~20 min
│   ├── Notebook_03_PhraseBank_Classification.ipynb     # T4 GPU, ~40 min
│   ├── Notebook_04_DigitLevel_Synthetic.ipynb          # T4 GPU, ~20 min
│   ├── nb_builder.py                # helper: builds .ipynb from Python
│   └── build_nb{1,2,3,4}.py         # source-of-truth for each notebook
├── results/
│   ├── notebook_1/   # fragmentation matrices, integer-range token counts
│   ├── notebook_2/   # magnitude- and comparison-probe summaries
│   ├── notebook_3/   # PhraseBank F1 matrix, per-subset breakdown
│   └── notebook_4/   # synthetic OOD accuracies
├── poster/
│   ├── poster.pdf    # final A1 landscape poster
│   └── build_poster.py
├── requirements.txt
├── LICENSE
└── README.md         # this file
```

---

## Reproducing the experiments

### Environment

Free-tier Colab / Kaggle T4 GPUs are sufficient. Local CPU is enough for Notebook 1.

```bash
pip install -r requirements.txt
```

### Order of execution

Run notebooks in numerical order:

| # | Notebook | Hardware | Time | Output |
|---|----------|----------|------|--------|
| 1 | `Notebook_01_Tokenizer_Fragmentation.ipynb` | CPU | ~5 min | `results/notebook_1/*.csv` |
| 2 | `Notebook_02_Number_Representation_Probing.ipynb` | 1× T4 | ~20 min | `results/notebook_2/*.csv` |
| 3 | `Notebook_03_PhraseBank_Classification.ipynb` | 1× T4 | ~40 min | `results/notebook_3/*.csv` + `phrasebank_results.json` |
| 4 | `Notebook_04_DigitLevel_Synthetic.ipynb` | 1× T4 | ~20 min | `results/notebook_4/*.csv` + `nb4_comparison.json` |

Notebooks are self-contained — they can be run independently once the dependencies are installed.

### Rebuilding the poster

```bash
python poster/build_poster.py
```

The script reads the CSV outputs from `results/notebook_{1..4}/` and regenerates `poster/poster.pdf`.

---

## Experimental design

### Notebook 1 — Tokenizer fragmentation (RQ 1)

We tokenize a categorized corpus of numeric expressions (plain/comma integers, decimals, scientific notation, dates, times, currency, percentages, fractions) with **9 tokenizers** spanning five families:

- **WordPiece**: `bert-base-uncased`
- **Byte-level BPE**: `gpt2`, `roberta-base`, `facebook/opt-125m`, `bigscience/bloom-560m`
- **SentencePiece**: `t5-base`, `xlm-roberta-base`
- **GPT-NeoX BPE**: `EleutherAI/pythia-160m`
- **Qwen2 byte-BPE (digit-level)**: `Qwen/Qwen2.5-0.5B` — our real digit-level reference

We measure mean sub-tokens per expression per category, single-token coverage for integers 0–9999, and sequence-length inflation on real finance/science sentences.

**Empirical finding:** Qwen2.5 produces 4.88 sub-tokens per small integer (≈ mean number of digits) while Pythia-160m produces only 1.44 — confirming that Qwen2.5's tokenizer is digit-level in practice and Pythia-160m's is not, despite both being in the GPT-NeoX/byte-BPE lineage.

### Notebook 2 — Number-representation probing (RQ 2)

For **6 comparable models** (BERT, RoBERTa, GPT-2, Pythia-160m, T5-small, **Qwen2.5-0.5B**), we:

1. Feed `str(n)` to the model, mean-pool sub-token embeddings from the last hidden state.
2. Fit ridge regression `embedding → log₁₀(n)` on integers 1–1000.
3. Evaluate R² on a held-out in-distribution slice and on an OOD range (10k–100k).

We also run a number-comparison probe: a logistic classifier over concatenated embeddings `[emb(a); emb(b)]` predicting whether `a > b`.

**Empirical finding:** Qwen2.5's in-distribution R² is lower (0.67 vs. 0.89–0.97 for others) because digit-level representations spread information across multiple tokens, and mean-pooling dilutes it. Yet its comparison accuracy stays strong (0.91) — pair-level information survives the pooling. Critically, **all 6 models have strongly negative OOD R²**. Digit-level tokenization alone does not produce magnitude-aware embeddings.

### Notebook 3 — Financial PhraseBank (RQ 3)

We use the `sentences_50agree` subset (~4840 sentences, 3-class sentiment).

To bypass the HuggingFace dataset-script deprecation in `datasets ≥ 4.0`, we download the canonical `FinancialPhraseBank-v1.0.zip` directly from the dataset repo via `huggingface_hub.hf_hub_download` and parse the txt files ourselves.

We fine-tune 3 encoders (DistilBERT, RoBERTa, XLM-R) under 3 preprocessing conditions (9 runs total, 2 epochs each, fixed hyperparameters):

- `original`: text as-is
- `mask_nums`: every numeric token replaced with `[NUM]`
- `digit_split`: every digit separated by a space (`1234` → `1 2 3 4`)

### Notebook 4 — Synthetic OOD number comparison (RQ 4)

Binary classifiers predict `a > b` given the string `"{a} {b}"`:

- Training pairs from `[0, 1000)`.
- Evaluation: in-distribution `[0, 1000)` + OOD `[10_000, 100_000)`.
- Conditions: DistilBERT plain, DistilBERT + digit-split preprocessing, Pythia-160m.

This is the cleanest probe of compositional generalization to unseen magnitudes — and where the argument for digit-level tokenization is strongest.

---

## Key results (real numbers)

### Fragmentation (mean sub-tokens per expression)

| Tokenizer    | Plain int. | Commas | Decimals | Scientific | Dates |
|--------------|-----------:|-------:|---------:|-----------:|------:|
| BERT         | 2.5 | 4.2 | 3.7 | 4.9 | 5.6 |
| GPT-2        | 1.8 | 4.1 | 3.5 | 4.8 | 5.3 |
| RoBERTa      | 1.8 | 4.1 | 3.5 | 4.8 | 5.3 |
| BLOOM        | 1.4 | 4.0 | 3.3 | 3.8 | 4.1 |
| T5           | 2.2 | 2.4 | 2.6 | 3.8 | 4.3 |
| XLM-R        | 1.8 | 2.3 | 2.1 | 3.6 | 3.9 |
| Pythia       | 1.4 | 4.0 | 3.3 | 4.8 | 5.6 |
| OPT          | 1.8 | 4.1 | 3.5 | 4.8 | 5.3 |
| **Qwen2.5**  | **4.9** | **7.7** | **5.3** | **5.7** | **9.7** |

### Number-representation probing

| Model    | R²ᵢₙ | R²ₒₒ𝒹 | Comparison acc |
|----------|------:|-------:|---------------:|
| BERT     | 0.97 | −40.6 | 0.957 |
| RoBERTa  | 0.96 | −44.7 | 0.924 |
| GPT-2    | 0.95 | −14.8 | 0.952 |
| Pythia   | 0.89 | −25.9 | 0.909 |
| T5       | 0.97 | −35.0 | 0.947 |
| **Qwen2.5** | **0.67** | **−20.4** | **0.913** |

### PhraseBank macro-F1 (test)

| Model      | original | mask_nums | digit_split |
|------------|---------:|----------:|------------:|
| DistilBERT | 0.761 | 0.752 | 0.754 |
| RoBERTa    | 0.810 | 0.810 | 0.797 |
| XLM-R      | 0.478 | 0.569 | 0.543 |

### Synthetic comparison task accuracy

| Condition                    | in-dist | OOD   | mean seq len |
|------------------------------|--------:|------:|-------------:|
| DistilBERT · plain           | 0.987   | 0.758 | 5.1 |
| **DistilBERT · digit_split** | **0.992** | **0.899** | **8.8** |
| Pythia-160m · native         | 0.751   | 0.602 | 2.5 |

---

## References

- Malo, Sinha, Takala, Korhonen, Wallenius. *"Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts."* JASIST, 2014.
- Wallace, Wang, Li, Singh, Gardner. *"Do NLP Models Know Numbers? Probing Numeracy in Embeddings."* EMNLP, 2019.
- Thawani, Pujara, Ilievski, Szekely. *"Representing Numbers in NLP: A Survey and a Vision."* NAACL, 2021.
- Yang, Yang et al. *"Qwen2.5 Technical Report."* Alibaba, 2024. [Confirms per-digit pre-tokenization.]
- Dubey et al. *"The Llama 3 Herd of Models."* Meta AI, 2024.

---

## License

Code: MIT. Financial PhraseBank is CC-BY-NC-SA-3.0 (retained from original authors).
