# dnalm-hub development tasks. Deployment is plain `docker compose` (see compose.yaml).
#
# One service per model family under services/, shared code under packages/. Each
# has its OWN environment (pyproject.toml + uv.lock) because model families need
# incompatible dependencies (e.g. NTv2 needs transformers<5, NTv3 runs 5.x), so
# there is no single venv -- these targets loop over the projects.
#
#   make help                      list targets
#   make test [S=ntv2]             run tests (all packages + services, or one service)
#   make lock                      re-lock every project after editing a pyproject.toml
#   make build [S=ntv2]            build GPU image(s) without starting them
#   make new-service NAME=hyenadna scaffold a new model family from services/_template

UV ?= uv
SERVICES_DIR ?= services
SERVICES := $(sort $(filter-out _template,$(notdir $(wildcard services/*))))
PACKAGES := $(sort $(notdir $(wildcard packages/*)))
S ?= $(SERVICES)
S_GIVEN := $(filter command line environment,$(origin S))

.PHONY: help list lock test build new-service

help:
	@sed -n 's/^#   //p' $(MAKEFILE_LIST)
	@echo; echo "services: $(SERVICES)"; echo "packages: $(PACKAGES)"
	@echo; echo "deploy:   docker compose up -d --build | ps | logs -f <service> | down"

list:
	@for s in $(SERVICES); do echo "$$s	services/$$s	$$(grep -m1 '^description' services/$$s/pyproject.toml | cut -d'"' -f2)"; done

lock:
	@for d in $(addprefix packages/,$(PACKAGES)) $(addprefix services/,$(SERVICES)); do \
		echo "== uv lock: $$d"; (cd $$d && $(UV) lock) || exit 1; done

test:
	@set -e; for d in $(if $(S_GIVEN),$(addprefix services/,$(S)),$(addprefix packages/,$(PACKAGES)) $(addprefix services/,$(SERVICES))); do \
		echo "== pytest: $$d"; (cd $$d && $(UV) run pytest -q -p no:cacheprovider tests); done

build:
	@for s in $(S); do echo "== docker build: $$s-mcp:gpu"; \
		docker build -f services/$$s/Dockerfile -t $$s-mcp:gpu . || exit 1; done

new-service:
	@test -n "$(NAME)" || { echo "usage: make new-service NAME=hyenadna"; exit 1; }
	@echo "$(NAME)" | grep -Eq '^[a-z][a-z0-9]*$$' || { echo "NAME must be lowercase letters/digits, e.g. hyenadna, dnabert2"; exit 1; }
	@test ! -e $(SERVICES_DIR)/$(NAME) || { echo "$(SERVICES_DIR)/$(NAME) already exists"; exit 1; }
	cp -r services/_template $(SERVICES_DIR)/$(NAME)
	mv $(SERVICES_DIR)/$(NAME)/src/__PKG__ $(SERVICES_DIR)/$(NAME)/src/$(NAME)_mcp
	grep -rl '__NAME__\|__PKG__\|__ENV__' $(SERVICES_DIR)/$(NAME) | xargs sed -i \
		-e 's/__PKG__/$(NAME)_mcp/g' -e 's/__NAME__/$(NAME)/g' -e "s/__ENV__/$$(echo $(NAME) | tr a-z A-Z)/g"
	cd $(SERVICES_DIR)/$(NAME) && $(UV) lock
	@echo; echo "Created $(SERVICES_DIR)/$(NAME). Next steps: docs/adding-a-model.md"
