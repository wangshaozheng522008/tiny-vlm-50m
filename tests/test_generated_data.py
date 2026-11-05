import hashlib

from PIL import Image

from scripts.gen_multimodal import QUESTIONS, generate_dataset
from tiny_vlm.data import read_records


def test_generated_labels_and_no_cross_split_leakage(tmp_path):
    meta = generate_dataset(tmp_path, train_scenes=3, val_scenes=2, test_scenes=2, size=32)
    seen_images, seen_text = set(), set()
    for split in ("train", "val", "test"):
        records = read_records(tmp_path / f"{split}.jsonl")
        assert set(r.mode for r in records) == {"text", "image", "mixed"}
        images = {r.image for r in records if r.image is not None}
        digests = {hashlib.sha256(Image.open(path).tobytes()).hexdigest() for path in images}
        prompts = {r.prompt for r in records if r.image is None}
        assert not seen_images & digests
        assert not seen_text & prompts
        assert digests == set(meta["splits"][split]["image_hashes"])
        seen_images.update(digests)
        seen_text.update(prompts)
        for record in records:
            if record.mode == "image":
                # Identical image-only inputs have one canonical caption language.
                assert record.answer.endswith("。")
                count = record.answer[0]
                related = [r for r in records if r.image == record.image and r.mode == "mixed"]
                assert len(related) == 6
                counts = [r.answer for r in related if r.prompt in
                          (QUESTIONS["zh"][2], QUESTIONS["en"][2])]
                assert counts == [count, count]
                for language in QUESTIONS:
                    assert {r.prompt for r in related} >= set(QUESTIONS[language])


def test_generation_is_reproducible_and_english_mode_works(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    a = generate_dataset(left, 2, 1, 1, 32, seed=9, language="en")
    b = generate_dataset(right, 2, 1, 1, 32, seed=9, language="en")
    assert a == b
    for split in ("train", "val", "test"):
        assert (left / f"{split}.jsonl").read_bytes() == (right / f"{split}.jsonl").read_bytes()
        for record in read_records(left / f"{split}.jsonl"):
            assert record.answer.isascii()
