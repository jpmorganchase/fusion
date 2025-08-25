.PHONY: cross-perf, gha-dev

BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD)
REPO := jpmorganchase/fusion
WORKFLOW_FILE := dev.yml
PYTHON_VERSIONS := 3.9 3.10 3.11 3.12 3.13
LD_LIBRARY_PATHS := $(HOME)/.rye/py/cpython@3.8.18/lib:$(HOME)/.rye/py/cpython@3.9.18/lib:$(HOME)/.rye/py/cpython@3.10.14/lib:$(HOME)/.rye/py/cpython@3.11.9/lib:$(HOME)/.rye/py/cpython@3.12.3/lib:$(HOME)/.rye/py/cpython@3.13.0/lib

cross-perf:
	@for version in $(PYTHON_VERSIONS); do \
		rye pin $$version && \
		rye sync && \
		LD_LIBRARY_PATH=$(LD_LIBRARY_PATHS):$$LD_LIBRARY_PATH RUSTFLAGS="$(RUSTFLAGS)" maturin develop -r && \
		rye run pytest --color=yes --benchmark-only --benchmark-autosave || true; \
	done

cross-exp-perf:
	@for version in $(PYTHON_VERSIONS); do \
		rye pin $$version && \
		rye sync && \
		. .venv/bin/activate && \
		LD_LIBRARY_PATH=$(LD_LIBRARY_PATHS):$$LD_LIBRARY_PATH RUSTFLAGS="$(RUSTFLAGS)" maturin develop -r --features experiments && \
		rye run pytest --color=yes --benchmark-only --benchmark-autosave --experiments || true; \
	done

gha-dev:
	gh workflow run $(WORKFLOW_FILE) --ref $(BRANCH_NAME) --repo $(REPO)

gha-dev-cancel:
	gh run list --branch $(BRANCH_NAME) --workflow=$(WORKFLOW_FILE) --limit 1 --json databaseId --jq '.[].databaseId' | xargs -I {} gh run cancel {}