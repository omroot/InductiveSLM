

def to_text(ex: dict[str, str]) -> dict[str, str]:
    obs = ex["Training Observations"].strip()
    q = ex["Question"].strip()
    a = ex["Answer"].strip()
    prompt = f"Training Observations:\n{obs}\n\nQuestion:\n{q}\n\nAnswer:\n"
    return {"prompt": prompt, "response": a, "text": prompt + a}

