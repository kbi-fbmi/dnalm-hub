# dnalm-hub development tasks. Deployment is plain `docker compose` (see compose.yaml).
#
# One service per model family under services/, shared code under packages/. Each
# has its OWN environment (pyproject.toml + uv.lock) because model families need
# incompatible dependencies (e.g. NTv2 needs transformers<5, NTv3 runs 5.x), so
# there is no single venv -- these targets loop over the projects.
#
#   make help                      list targets
#   make list                      services with their descriptions
#   make sync [S=ntv2] [CLEAN=1]   create/update venvs from uv.lock (CLEAN=1 recreates them, e.g. after moving the repo)
#   make test [S=ntv2]             run tests (all packages + services, or one service)
#   make lint                      ruff check + format check over the whole repo
#   make format                    ruff autofix + format over the whole repo
#   make lock                      re-lock every project after editing a pyproject.toml
#   make build [S=ntv2]            build GPU image(s) defined in compose.yaml, without starting them
#   make new-service NAME=hyenadna scaffold a new model family from services/_template
#   make smoke [S=ntv2]            smoke-test a running stack (all services + gateway; MCP_HOST, MCP_AUTH_TOKEN)
#   make version                   print the repo version (one version for everything)
#   make check-version [TAG=v0.9.0] check every pyproject/uv.lock/CITATION.cff agrees (and matches TAG)
#   make bump VERSION=0.9.1        set a new version everywhere (then: CHANGELOG.md, commit, tag)

UV ?= uv
RUFF ?= uvx ruff@0.16.10
SERVICES_DIR ?= services
TEMPLATE := services/_template
SERVICES := $(sort $(filter-out _template,$(notdir $(wildcard $(SERVICES_DIR)/*))))
PACKAGES := $(sort $(notdir $(wildcard packages/*)))
S ?= $(SERVICES)
S_GIVEN := $(filter command line environment,$(origin S))
# Packages + services, or only the services named in S.
PROJECTS := $(if $(S_GIVEN),$(addprefix $(SERVICES_DIR)/,$(S)),$(addprefix packages/,$(PACKAGES)) $(addprefix $(SERVICES_DIR)/,$(SERVICES)))

# An activated venv (pyenv, conda, ...) would make uv warn in every project; each
# project uses its own .venv anyway.
unexport VIRTUAL_ENV

.PHONY: help list sync lock test lint format build new-service smoke version check-version bump

help:
	@sed -n 's/^#   //p' $(MAKEFILE_LIST)
	@echo; echo "services: $(SERVICES)"; echo "packages: $(PACKAGES)"
	@echo; echo "deploy:   docker compose up -d --build | ps | logs -f <service> | down"

list:
	@for s in $(SERVICES); do echo "$$s	$(SERVICES_DIR)/$$s	$$(grep -m1 '^description' $(SERVICES_DIR)/$$s/pyproject.toml | cut -d'"' -f2)"; done

sync:
	@set -e; for d in $(PROJECTS); do \
		echo "== uv sync: $$d"; $(if $(CLEAN),rm -rf $$d/.venv;) (cd $$d && $(UV) sync --frozen); done

lock:
	@set -e; for d in $(addprefix packages/,$(PACKAGES)) $(addprefix $(SERVICES_DIR)/,$(SERVICES)); do \
		echo "== uv lock: $$d"; (cd $$d && $(UV) lock); done

test:
	@set -e; for d in $(PROJECTS); do \
		echo "== pytest: $$d"; (cd $$d && $(UV) run pytest -q -p no:cacheprovider tests); done

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) check --fix .
	$(RUFF) format .

# compose.yaml is the single source of truth for image names and build settings.
build:
	docker compose build $(if $(S_GIVEN),$(S))

new-service:
	@test -n "$(NAME)" || { echo "usage: make new-service NAME=hyenadna"; exit 1; }
	@echo "$(NAME)" | grep -Eq '^[a-z][a-z0-9]*$$' || { echo "NAME must be lowercase letters/digits, e.g. hyenadna, dnabert2"; exit 1; }
	@test ! -e $(SERVICES_DIR)/$(NAME) || { echo "$(SERVICES_DIR)/$(NAME) already exists"; exit 1; }
	cp -r $(TEMPLATE) $(SERVICES_DIR)/$(NAME)
	mv $(SERVICES_DIR)/$(NAME)/src/__PKG__ $(SERVICES_DIR)/$(NAME)/src/$(NAME)_mcp
	grep -rl '__NAME__\|__PKG__\|__ENV__' $(SERVICES_DIR)/$(NAME) | xargs sed -i \
		-e 's/__PKG__/$(NAME)_mcp/g' -e 's/__NAME__/$(NAME)/g' -e "s/__ENV__/$$(echo $(NAME) | tr a-z A-Z)/g"
	$(RUFF) format -q $(SERVICES_DIR)/$(NAME)
	cd $(SERVICES_DIR)/$(NAME) && $(UV) lock
	@echo; echo "Created $(SERVICES_DIR)/$(NAME). Next steps (docs/adding-a-model.md):"
	@echo "  1. implement the model in $(SERVICES_DIR)/$(NAME)/src/$(NAME)_mcp/"
	@echo "  2. add a '$(NAME)' block to compose.yaml (new host port) and add it to the gateway's GATEWAY_BACKENDS"
	@echo "  3. make build S=$(NAME) && docker compose up -d $(NAME)"

version:
	@python3 scripts/version.py

check-version:
	@python3 scripts/version.py check $(if $(TAG),--tag $(TAG))

bump:
	@test -n "$(VERSION)" || { echo "usage: make bump VERSION=0.9.1"; exit 1; }
	python3 scripts/version.py set $(VERSION)

# Against a running deployment (docker compose up -d); reads MCP_HOST / MCP_AUTH_TOKEN.
smoke:
	cd packages/dnalm-client && $(UV) run python ../../scripts/smoke_test.py $(if $(S_GIVEN),$(S))
