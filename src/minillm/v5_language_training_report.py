"""Render the observed V5 training and evaluation curves as standalone SVGs."""

import csv
from html import escape


def knowledge_score(result):
    """Count exact automatically scored geography/history and science responses."""
    categories = result["generation"][0]["benchmark"]["scores"]["raw"]["categories"]
    knowledge = [categories[name] for name in ("geohistory", "science")]
    return sum(row["correct"] for row in knowledge), sum(row["scored"] for row in knowledge)


def _chart(path, title, series):
    width, height = 850, 310
    left, right, top, bottom = 83, width - 24, 43, height - 65
    x_max = max(x for _, points in series for x, _ in points)
    numbers = [y for _, points in series for _, y in points]
    lo, hi = min(numbers), max(numbers)
    padding = (hi - lo) * .08 or max(abs(hi) * .08, .01)
    lo, hi = lo - padding, hi + padding

    def coord(x, y):
        return left + (right - left) * x / x_max, bottom - (bottom - top) * (y - lo) / (hi - lo)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img"><title>{escape(title)}</title>',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<g font-family="sans-serif" font-size="12" fill="#222"><text x="{left}" y="25" '
             f'font-size="17">{escape(title)}</text>']
    for i in range(6):
        y = lo + (hi - lo) * i / 5
        _, yy = coord(0, y)
        x = x_max * i / 5
        xx, _ = coord(x, lo)
        parts.append(f'<path d="M{left} {yy:.2f}H{right}" stroke="#e1e4eb"/>')
        parts.append(f'<text x="{left - 8}" y="{yy + 4:.2f}" text-anchor="end">{y:.4g}</text>')
        parts.append(f'<text x="{xx:.2f}" y="{bottom + 20}" text-anchor="middle">{x / 1e6:.2f}M</text>')
    parts.append(f'<text x="{(left + right) / 2}" y="{bottom + 39}" text-anchor="middle">Effective supervised tokens</text>')
    for i, (label, points) in enumerate(series):
        color = ("#1665b8", "#9b3d27", "#267348")[i]
        coordinates = [coord(x, y) for x, y in points]
        vertices = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.7" points="{vertices}"/>')
        if len(points) <= 12:
            parts.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>' for x, y in coordinates)
        parts.append(f'<text x="{left + i * 230}" y="{height - 8}" fill="{color}">{escape(label)}</text>')
    path.write_text("\n".join([*parts, "</g></svg>"]) + "\n", encoding="utf-8")


def plot_results(directory, results):
    p = directory / "loss-and-learning-rate.csv"
    with p.open(newline="", encoding="utf-8") as handle:
        observations = list(csv.DictReader(handle))
    nodes = [result for name, result in results.items() if name.startswith("step") and name[4:].isdigit()]

    def observed(field):
        return [(int(row["tokens_seen"]), float(row[field])) for row in observations if row[field]]

    def metric(extract):
        return [(row["tokens_seen"], extract(row)) for row in nodes]

    def greedy(row):
        return row["generation"][0]["benchmark"]

    charts = {
        "loss": ("Training and validation loss", [
            ("Train update", observed("train_loss")), ("Fixed Train", metric(lambda row: row["lm"]["train"]["loss"])),
            ("V5 Val", metric(lambda row: row["lm"]["v5_val"]["loss"]))]),
        "ppl": ("Fixed-window perplexity", [
            ("Common Val", metric(lambda row: row["lm"]["val"]["perplexity"])),
            ("V5 Val", metric(lambda row: row["lm"]["v5_val"]["perplexity"]))]),
        "repetition": ("Greedy repetition", [
            ("Token 4-gram", metric(lambda row: greedy(row)["mean_token_4gram_repeat_ratio"])),
            ("Character 3-gram", metric(lambda row: greedy(row)["mean_character_3gram_repeat_ratio"]))]),
        "han": ("Greedy outputs containing Han characters", [
            ("Han output fraction", metric(lambda row: greedy(row)["han_output_fraction"]))]),
        "knowledge": ("Greedy geography/history and science accuracy", [
            ("Automatically scored", metric(lambda row: knowledge_score(row)[0] / knowledge_score(row)[1]))]),
        "learning-rate": ("Observed learning rate", [("AdamW learning rate", observed("observed_learning_rate"))]),
        "gradient-norm": ("Observed gradient norm", [("Before gradient clipping", observed("gradient_norm"))]),
    }
    for name, (title, series) in charts.items():
        _chart(directory / f"{name}.svg", title, series)
