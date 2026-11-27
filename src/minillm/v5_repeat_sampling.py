"""Replay committed and planned packed-window consumption across V5 epochs."""

from collections import Counter
from dataclasses import dataclass

from .corpus_utils import identity
from .data import EpochBatchSampler
from .v4_training_accounting import window_token_counts
from .v5_language_corpus import V5_CATEGORY_RATIOS, V5_TOLERANCE, _caps, _source_mix


@dataclass(frozen=True)
class RepeatPlan:
    sampler: EpochBatchSampler
    parent_windows: int
    added_windows: int

    def __post_init__(self):
        if (self.sampler.batch_size != 1 or not 0 < self.parent_windows < len(self.sampler)
                or not 0 < self.added_windows <= 2 * len(self.sampler) - self.parent_windows):
            raise ValueError("repeat mode requires one frozen epoch and a bounded second epoch")

    def cursor(self, added):
        if not 0 <= added <= self.added_windows:
            raise ValueError("repeat sampler cursor exceeds the authorized stage")
        return divmod(self.parent_windows + added, len(self.sampler))

    def epochs(self, start, stop):
        """Yield actual EpochBatchSampler slices for a cumulative window interval."""
        if not 0 <= start <= stop <= self.parent_windows + self.added_windows:
            raise ValueError("repeat sampler interval exceeds the authorized stage")
        while start < stop:
            epoch, offset = divmod(start, len(self.sampler))
            count = min(stop - start, len(self.sampler) - offset)
            windows = self.sampler.indices(epoch)[offset:offset + count]
            if len(windows) != count or len(set(windows)) != count:
                raise ValueError(f"epoch {epoch} repeated or omitted windows")
            yield epoch, offset, windows
            start += count

    def stage_order(self):
        return [window for _, _, windows in self.epochs(
            self.parent_windows, self.parent_windows + self.added_windows) for window in windows]


def accounting(replay, documents, corpus_plan, seq_len, added, *, enforce=False):
    categories = {source["source_id"]: source["category"] for source in corpus_plan["registry"]["sources"]}
    caps = _caps(corpus_plan["v5_inputs"]["candidate_policy"])
    cumulative = replay.parent_windows + added
    tokens_per_window = seq_len - 1

    def measure(start, stop):
        category_counts, source_counts = Counter(), Counter()
        unique, epochs = set(), []
        ordered = []
        for epoch, offset, windows in replay.epochs(start, stop):
            epochs.append({"epoch": epoch, "start_window": offset, "stop_window": offset + len(windows),
                           "windows": len(windows), "coverage": len(windows) / len(replay.sampler)})
            ordered.extend((epoch, window) for window in windows)
            unique.update(windows)
            category_counts.update(window_token_counts(documents, seq_len, windows, "category"))
            source_counts.update(window_token_counts(documents, seq_len, windows, "source_id"))
        total = (stop - start) * tokens_per_window
        if sum(category_counts.values()) != total or sum(source_counts.values()) != total:
            raise ValueError("repeat sampler supervision disagrees with the frozen token index")
        mix = {name: category_counts[name] / total for name in V5_CATEGORY_RATIOS}
        sources = _source_mix(source_counts, categories, caps)
        mix_ok = all(abs(mix[name] - ratio) <= V5_TOLERANCE if ratio else category_counts[name] == 0
                     for name, ratio in V5_CATEGORY_RATIOS.items())
        sources_ok = all(source["within_cap"] for source in sources.values())
        if enforce and not (mix_ok and sources_ok):
            raise ValueError("repeat sampler violates category ratios or source caps")
        return {"windows": stop - start, "effective_tokens": total,
                "unique_effective_tokens": len(unique) * tokens_per_window,
                "repeated_effective_tokens": total - len(unique) * tokens_per_window,
                "ordered_epoch_windows_sha256": identity(ordered), "epochs": epochs,
                "category_tokens": {name: category_counts[name] for name in V5_CATEGORY_RATIOS},
                "category_shares": mix, "category_mix_ok": mix_ok, "sources": sources,
                "source_caps_ok": sources_ok}

    stage, total = measure(replay.parent_windows, cumulative), measure(0, cumulative)
    total_unique = total["unique_effective_tokens"]
    stage["new_effective_tokens"] = total_unique - replay.parent_windows * tokens_per_window
    stage["previously_consumed_effective_tokens"] = stage["effective_tokens"] - stage["new_effective_tokens"]
    return {"seed": replay.sampler.seed, "frozen_windows": len(replay.sampler),
            "stage": stage, "cumulative": total}
