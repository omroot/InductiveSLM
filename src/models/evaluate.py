import evaluate


def eval_metrics(preds: list[str], 
                 refs: list[str],
                 rouge: evaluate,
                 bleu: evaluate) -> dict[str, float]:
    try:
        r = rouge.compute(predictions=preds, references=refs)
        b = bleu.compute(predictions=preds, references=[[x] for x in refs])
        return {
            "rouge1": r.get("rouge1", 0.0),
            "rouge2": r.get("rouge2", 0.0),
            "rougeL": r.get("rougeL", 0.0),
            "rougeLsum": r.get("rougeLsum", 0.0),
            "bleu": b["score"]
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0, "bleu": 0.0}

def print_compare(baseline: dict[str, float], finetuned: dict[str, float]):
    print("\n===== OOS Metrics (Baseline vs Fine-tuned) =====")
    keys = ["rouge1", "rouge2", "rougeL", "rougeLsum", "bleu"]
    for k in keys:
        print(f"{k:10s}  base: {baseline[k]:6.3f}   ft: {finetuned[k]:6.3f}   Δ: {finetuned[k]-baseline[k]:+6.3f}")




