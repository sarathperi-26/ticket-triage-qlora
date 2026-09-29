# Consumer Complaint Triage with QLoRA

Fine-tunes a small open LLM (Qwen2.5-1.5B-Instruct) with QLoRA to read a raw consumer complaint and return strict JSON with a `product` and an `issue` label. The notebook benchmarks three systems on the same held-out test set: a majority-class guess, the prompted base model, and the fine-tuned model.

The full run, code, and outputs are in [`Fine-Tuned_LLM_for_Customer_Complaint_Triage__QLoRA_.ipynb`](Fine-Tuned_LLM_for_Customer_Complaint_Triage__QLoRA_.ipynb). Every cell's output is saved, so the results below can be verified directly in the notebook.

## Results

Evaluated on 300 held-out complaints from the CFPB Consumer Complaint Database. Single training run, seed 42.

| Metric | Majority guess | Prompted base | Fine-tuned (QLoRA) |
|---|---|---|---|
| Exact match (product and issue) | 1.0% | 8.0% | **58.0%** |
| Product accuracy | 19.0% | 50.0% | **84.0%** |
| Product macro-F1 | 0.053 | 0.474 | **0.837** |
| Issue accuracy | 29.7% | 14.0% | **60.0%** |
| Issue macro-F1 | 0.035 | 0.107 | **0.484** |
| Schema-valid outputs | 100.0% | 91.0% | **99.7%** |
| Seconds per example | n/a | **0.110** | 0.172 |

Notes on reading the table:

- Schema-valid means the output parsed as JSON and both labels come from the allowed lists. The base model wraps answers in markdown code fences, which the parser strips, so raw valid-JSON rate (100% for all three) flatters it. Schema-valid is the stricter metric.
- With 300 test examples, each accuracy figure carries roughly ±5 points of sampling error.
- The fine-tuned model is slower per example because the LoRA adapter is not merged into the base weights. No latency improvement is claimed.

Full metrics and error-analysis tables are in [`results/`](results/).

## Method

- **Data:** complaints with written narratives from a CFPB Narratives Archive export (November 2022 to August 2023). Narratives are cut to 1,000 characters. Renamed CFPB products are merged into six canonical products, and issues are reduced to the 12 most common plus `Other`.
- **Splits:** 3,000 train, 200 validation, 300 test. Each product is capped so credit reporting does not dominate, so the product mix is class-balanced and does not reflect real CFPB proportions.
- **Model:** Qwen2.5-1.5B-Instruct loaded in 4-bit NF4 with double quantization. LoRA rank 16, alpha 32, dropout 0.05, on all attention and MLP projection layers (18.5M trainable parameters, 1.18% of the model).
- **Training:** 2 epochs (376 steps), effective batch size 16, learning rate 2e-4 with cosine decay, fp16, paged 8-bit AdamW. Loss is computed only on the JSON answer tokens, not the prompt.
- **Hardware:** one NVIDIA L4 on Google Colab. Training took 13 minutes 36 seconds.
- **Validation loss:** 0.069 after epoch 1 and 0.063 after epoch 2. The test set was used only for the final comparison.

## Limitations

- Results come from one run with one seed on a 300-example test set.
- The three rarest issue types (66 to 98 training examples each) were never predicted correctly (0 of 8 each on the test set). Half of those errors defaulted to the `Other` catch-all and the rest went to adjacent labels, which suggests both class imbalance and overlapping label definitions.
- About 32% of issue labels fall in the `Other` catch-all, which caps how well issue accuracy can be measured.
- Checking or savings and money transfer complaints are confused in both directions.
- One test output used a product label outside the allowed list (`Health insurance`).
- The class-balanced sample is not representative of real complaint volumes.

## Run it

Open the notebook in Google Colab with a GPU runtime (developed on an NVIDIA L4). Run the cells in order, top to bottom.

One setup note: after the first cell installs `bitsandbytes`, restart the Colab session before continuing, otherwise the 4-bit model loading step will fail with an import error.

## Data

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/), public data. Complaint narratives are redacted by the CFPB (personal details appear as `XXXX`).

## License

MIT License.
