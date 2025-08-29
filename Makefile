.PHONY: gha-dev

BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD)
REPO := jpmorganchase/fusion
WORKFLOW_FILE := dev.yml

gha-dev:
	gh workflow run $(WORKFLOW_FILE) --ref $(BRANCH_NAME) --repo $(REPO)

gha-dev-cancel:
	gh run list --branch $(BRANCH_NAME) --workflow=$(WORKFLOW_FILE) --limit 1 --json databaseId --jq '.[].databaseId' | xargs -I {} gh run cancel {}