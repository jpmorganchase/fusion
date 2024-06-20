.PHONY: cross-perf, gha-dev

BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD)
REPO := jpmorganchase/fusion
WORKFLOW_FILE := dev.yml

cross-perf:
	@for version in 3.9 3.10 3.11 3.12; do \
		rye pin $$version && \
		rye sync && \
		maturin develop -r && \
		PATH=.venv/bin:$$PATH \
		LD_LIBRARY_PATH=$$HOME/.rye/py/cpython@3.8.18/lib:$$HOME/.rye/py/cpython@3.9.18/lib:$$HOME/.rye/py/cpython@3.10.14/lib:$$HOME/.rye/py/cpython@3.11.9/lib:$$HOME/.rye/py/cpython@3.12.3/lib:$$LD_LIBRARY_PATH \
		RUSTFLAGS="-L $$HOME/.rye/py/cpython@3.8.18/lib -L $$HOME/.rye/py/cpython@3.9.18/lib -L $$HOME/.rye/py/cpython@3.10.14/lib -L $$HOME/.rye/py/cpython@3.11.9/lib -L $$HOME/.rye/py/cpython@3.12.3/lib" \
		cargo test -r && \
		cargo llvm-cov report --html --output-dir .reports/rust && \
		pytest --color=yes --benchmark-only --benchmark-save=cross || true; \
	done

gha-dev:
	gh workflow run $(WORKFLOW_FILE) --ref $(BRANCH_NAME) --repo $(REPO)

gha-dev-cancel:
	gh run list --branch $(BRANCH_NAME) --workflow=$(WORKFLOW_FILE) --limit 1 --json databaseId --jq '.[].databaseId' | xargs -I {} gh run cancel {}