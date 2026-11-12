PYTHON ?= .venv/bin/python
DIAGNOSTIC_OUT ?= reports/diagnostics/nano-$(shell date +%Y%m%d-%H%M%S)
FOUNDATION_OUT ?= reports/diagnostics/chinese-$(shell date +%Y%m%d-%H%M%S)

.PHONY: diagnose-nano evaluate-chinese-foundation

diagnose-nano:
	$(PYTHON) scripts/diagnose_training.py --out $(DIAGNOSTIC_OUT) --seed 42 --max-steps 200 --max-effective-tokens 50000 --max-wall-seconds 120

# Explicit checkpoint/tokenizer are required. They are always read-only.
evaluate-chinese-foundation:
	@test -n "$(CKPT)" -a -n "$(TOKENIZER)" || (echo 'Set CKPT=... TOKENIZER=...'; exit 1)
	$(PYTHON) scripts/evaluate_chinese_foundation.py --ckpt "$(CKPT)" --tokenizer "$(TOKENIZER)" --out $(FOUNDATION_OUT) --seed 42 --max-wall-seconds 120
