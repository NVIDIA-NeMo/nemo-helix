# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Platform and architecture detection
UNAME_S := $(shell uname -s)
ARCH := $(shell uname -m)
PLATFORM := $(shell echo $(UNAME_S) | tr '[:upper:]' '[:lower:]')
PROFILE ?= platform
NMP_CONFIG_FILE_PATH ?= packages/nmp_platform/config/local.yaml

# Normalize architecture names and set arch-specific defaults.
ifeq ($(ARCH),x86_64)
	ARCH := amd64
	export BUILD_ARCH ?= linux/amd64
endif
ifeq ($(ARCH),aarch64)
	ARCH := arm64
endif
ifeq ($(ARCH),arm64)
	export BUILD_ARCH ?= linux/arm64
endif
PYTEST_EXTRA ?=
# Default development toolchain versions. Keep these aligned with Flox; the
# version-consistency checks validate them in pre-commit and CI.
PYTHON_VERSION ?= 3.12
UV_VERSION := 0.9.14
NODE_VERSION := 22.23.2
PNPM_VERSION := 10.34.5
GO_VERSION := 1.26.7
BOOTSTRAP_CREATE_VENV ?= 1
BOOTSTRAP_EXPECTED_VIRTUAL_ENV := $(CURDIR)/.venv
BOOTSTRAP_ACTIVATION_REMINDER = if [ "$(TOOLCHAIN)" = "flox" ] && [ "$${FLOX_ENV_PROJECT:-}" != "$(CURDIR)" ]; then echo ""; echo "Next steps:"; echo "  flox -q activate"; echo "  nemo --help"; elif [ "$(TOOLCHAIN)" = "system" ] && [ "$${VIRTUAL_ENV:-}" != "$(BOOTSTRAP_EXPECTED_VIRTUAL_ENV)" ]; then echo ""; echo "Next steps:"; echo "  source .venv/bin/activate"; echo "  nemo --help"; fi

# Display platform info
$(info local system architecture: $(PLATFORM)/$(ARCH))

# taken from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help:
	@echo "Makefile commands:"
	@grep -E '^[a-zA-Z_-][a-zA-Z0-9_/-]*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: toolchain-versions
toolchain-versions: ## Print the system toolchain versions accepted by bootstrap
	@printf 'Python %s\nuv >=%s\nNode.js %s\npnpm %s\nGo %s\n' "$(PYTHON_VERSION)" "$(UV_VERSION)" "$(NODE_VERSION)" "$(PNPM_VERSION)" "$(GO_VERSION)"

DOCKER_BAKE_FILE ?= docker-bake.hcl
DOCKER_TARGET ?= $(if $(TARGET),$(TARGET),docker-cpu)
DOCKER_PLATFORMS ?= $(BUILD_ARCH)
DOCKER_PLATFORM_SET = $(if $(DOCKER_PLATFORMS),--set "*.platform=$(DOCKER_PLATFORMS)",)
DOCKER_BAKE_ALLOW_FS_READ ?=
DOCKER_BAKE_ALLOW_FS_READ_FLAG = $(if $(DOCKER_BAKE_ALLOW_FS_READ), --allow=fs.read=$(DOCKER_BAKE_ALLOW_FS_READ),)

.PHONY: docker-list-targets
docker-list-targets: ## List Docker bake targets
	docker buildx bake -f $(DOCKER_BAKE_FILE) --list=targets

.PHONY: docker-print
docker-print: ## Print Docker bake graph for TARGET, default docker-cpu
	docker buildx bake -f $(DOCKER_BAKE_FILE) --print $(DOCKER_TARGET)

.PHONY: docker-build
docker-build: ## Build Docker bake TARGET without pushing, default docker-cpu
	docker buildx bake$(DOCKER_BAKE_ALLOW_FS_READ_FLAG) -f $(DOCKER_BAKE_FILE) $(DOCKER_TARGET)

.PHONY: docker-load
docker-load: ## Build and load single-platform Docker bake TARGET, default docker-cpu
	docker buildx bake$(DOCKER_BAKE_ALLOW_FS_READ_FLAG) -f $(DOCKER_BAKE_FILE) $(DOCKER_TARGET) $(DOCKER_PLATFORM_SET) --load

.PHONY: docker-push
docker-push: ## Build and push Docker bake TARGET, default docker-cpu
	docker buildx bake$(DOCKER_BAKE_ALLOW_FS_READ_FLAG) -f $(DOCKER_BAKE_FILE) $(DOCKER_TARGET) --push

.PHONY: refresh-openapi
refresh-openapi:  ## Generate the OpenAPI specification
	$(UV) run --frozen script/generate-openapi-spec.sh

.PHONY: stainless
stainless: ## Run Stainless to generate the OpenAPI spec and sync it with the SDK
	SDK_RELEASE_TIER=ga $(FLOX_EXEC) ./sdk/stainless.sh sync

.PHONY: generate
generate: stainless ## Alias for SDK generation via Stainless

.PHONY: update-web-sdk
update-web-sdk: verify-toolchain ## Regenerate the TypeScript web SDK (web/packages/sdk) from the OpenAPI spec via Orval
	cd web && $(PNPM) gen

.PHONY: update-sdk
update-sdk: build-policy refresh-openapi stainless update-web-sdk update-cli ## Update the SDK by regenerating the OpenAPI spec and syncing it with Stainless

.PHONY: vendor-nemo-platform-ext
vendor-nemo-platform-ext:
	$(FLOX_EXEC) $(MAKE) -C packages/nemo_platform_ext vendor

.PHONY: generate-cli-commands
generate-cli-commands: ## Run generation of the CLI commands
	$(UV) run --frozen nemo-platform-sdk-tools generate-cli $(ARGS)

	# auto-generated code can be cleaned up more aggressively (in this case, we want to remove unused imports in __init__.py files)
	$(UV) run --frozen ruff check --fix --preview --unsafe-fixes --extend-select F401,E402 packages/nemo_platform_ext/src/nemo_platform_ext/cli/commands/api/
	# ARG001 catches unused function arguments which indicates variable shadowing bugs (no auto-fix)
	$(UV) run --frozen ruff check --select ARG001 packages/nemo_platform_ext/src/nemo_platform_ext/cli/commands/api/
	$(UV) run --frozen ruff check --fix --unsafe-fixes packages/nemo_platform_ext/src/nemo_platform_ext/cli/commands/api/
	$(UV) run --frozen ruff format packages/nemo_platform_ext

.PHONY: generate-cli-reference-docs
generate-cli-reference-docs: ## Generate the CLI reference documentation
	$(UV) run --frozen packages/nemo_platform_ext/scripts/docs_generator.py reference > docs/cli/reference.mdx
	$(UV) run --frozen packages/nemo_platform_ext/scripts/docs_generator.py summary > docs/fern/snippets/_snippets/cli-summary.mdx

.PHONY: generate-config-reference-docs
generate-config-reference-docs: ## Generate the platform config reference documentation
	$(UV) run --frozen generate-config-docs

# ============================================================================
# Fern documentation site (docs/fern)
# ============================================================================
# Convenience wrappers around `cd docs/fern && npm run ...` so contributors
# don't have to remember the fern-api invocations. The CI workflows
# (.github/workflows/fern-docs-*.yaml) are the source of truth. See
# docs/fern/README.md and docs/AGENTS.md. First run on a machine:
# `make docs-deps` (and `make docs-login` once for the Fern org).

.PHONY: docs-deps
docs-deps: ## Install the Fern docs tooling (docs/fern node deps)
	cd docs/fern && npm ci

.PHONY: docs
docs: ## Start the Fern docs dev server (local preview, prints a localhost URL)
	cd docs/fern && npm run dev

.PHONY: docs-watch
docs-watch: ## Start Fern docs dev plus a repo-level watcher for docs/** changes
	cd docs/fern && npm run watch

.PHONY: docs-check
docs-check: ## Validate the Fern docs (fern check + validate-mdx + gated-link check)
	cd docs/fern && npm run check

.PHONY: docs-lint-snippets
docs-lint-snippets: ## Syntax-check Python fenced snippets across all docs
	$(UV) run --frozen python docs/_scripts/lint_python_snippets.py docs --no-type-check

.PHONY: docs-check-python-snippets
docs-check-python-snippets: ## Syntax-check and type-check Python snippets in one doc (DOCS_PATH=...)
	@if [ -z "$(strip $(DOCS_PATH))" ]; then echo "Usage: make docs-check-python-snippets DOCS_PATH=docs/customizer/tutorials/import-hf-model.mdx" >&2; exit 2; fi
	$(UV) run --frozen python docs/_scripts/lint_python_snippets.py "$(DOCS_PATH)"

.PHONY: docs-run-notebook
docs-run-notebook: ## Execute one Fern notebook source (DOCS_PATH=.mdx/.ipynb/.md, optional ARGS=...)
	@if [ -z "$(strip $(DOCS_PATH))" ]; then echo "Usage: make docs-run-notebook DOCS_PATH=docs/customizer/tutorials/sft-customization-job.mdx" >&2; exit 2; fi
	$(UV) run --frozen python docs/fern/scripts/run_notebooks.py $(ARGS) "$(DOCS_PATH)"

.PHONY: docs-broken-links
docs-broken-links: ## Report broken links across the built docs
	cd docs/fern && npm run broken-links

.PHONY: docs-fix-links
docs-fix-links: ## Delink references from published pages into gated (unready) pages
	cd docs/fern && npm run fix:gated-links

.PHONY: docs-preview
docs-preview: ## Build a shared Fern preview URL (needs DOCS_FERN_TOKEN)
	cd docs/fern && npm run preview

.PHONY: docs-login
docs-login: ## One-time Fern CLI auth (for the nvidia Fern org)
	npx -y fern-api@latest login

.PHONY: docs-publish
docs-publish: ## Trigger the Publish Fern Docs workflow (normally runs on push to main)
	gh workflow run publish-fern-docs.yaml

.PHONY: update-cli
update-cli: generate-cli-commands vendor-nemo-platform-ext generate-cli-reference-docs

.PHONY: clean-python
clean-python: ## remove python virtual environment
	rm -rf .venv/

.PHONY: verify-python-version
verify-python-version: verify-toolchain ## Verify Python version and install if necessary
	@echo "~~~~~~"
	@echo "verifying python version"
	$(UV) python find $(PYTHON_VERSION) || $(UV) python install $(PYTHON_VERSION)

# Phony rather than a file target: a `.venv/bin/python` that already exists says
# nothing about which interpreter it is, so changing PYTHON_VERSION has to force
# a rebuild. Order-only prerequisite keeps tool resolution out of the recipe.
.PHONY: .venv
.venv: | verify-python-version ## Create a Python virtual environment
	@echo "~~~"
	@current=$$([ -x .venv/bin/python ] && .venv/bin/python -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null); \
	case "$$current" in \
		"$(PYTHON_VERSION)"|"$(PYTHON_VERSION)".*) exit 0 ;; \
	esac; \
	if [ "$(BOOTSTRAP_CREATE_VENV)" = "0" ]; then \
		if [ -n "$$current" ]; then \
			echo "BOOTSTRAP_CREATE_VENV=0 but .venv is on Python $$current, not $(PYTHON_VERSION)"; \
		else \
			echo "BOOTSTRAP_CREATE_VENV=0 but .venv/bin/python is missing or not runnable"; \
		fi; \
		echo "Create .venv manually, or run make again without BOOTSTRAP_CREATE_VENV=0."; \
		exit 1; \
	fi; \
	if [ -n "$$current" ]; then \
		echo "recreating .venv on Python $(PYTHON_VERSION) (was $$current)"; \
	elif [ -d .venv ]; then \
		echo "replacing unusable .venv with Python $(PYTHON_VERSION)"; \
	else \
		echo "setting up a venv with uv"; \
	fi; \
	$(UV) venv --python $(PYTHON_VERSION) --seed $$([ -d .venv ] && echo --clear)

# Optional escape hatch for local plugin packages that cannot participate in the
# root uv workspace/lock. Leave empty for the normal monorepo bootstrap path.
BOOTSTRAP_LOCAL_PLUGIN_DIRS ?=

.PHONY: bootstrap-python
bootstrap-python: verify-python-version verify-compiler ## Bootstrap Python dependencies.
	@echo "~~~~~~"
	@echo "installing python dependencies"
	$(UV) sync --python $(PYTHON_VERSION) --frozen --all-packages
	@if [ -n "$(strip $(BOOTSTRAP_LOCAL_PLUGIN_DIRS))" ]; then \
		$(MAKE) bootstrap-plugins BOOTSTRAP_LOCAL_PLUGIN_DIRS="$(BOOTSTRAP_LOCAL_PLUGIN_DIRS)"; \
	fi
	@if [ "$(filter bootstrap-python,$(MAKECMDGOALS))" = "bootstrap-python" ]; then \
		$(BOOTSTRAP_ACTIVATION_REMINDER); \
	fi

TOOLCHAIN ?= flox
FLOX ?= flox
FLOX_QUIET ?= $(FLOX) -q
ifeq ($(TOOLCHAIN),flox)
ifeq ($(FLOX_ENV_PROJECT),$(CURDIR))
FLOX_EXEC :=
else
FLOX_EXEC := $(FLOX_QUIET) activate --dir "$(CURDIR)" --
endif
else ifeq ($(TOOLCHAIN),system)
FLOX_EXEC :=
else
$(error TOOLCHAIN must be either flox or system)
endif
UV := $(FLOX_EXEC) uv
ifeq ($(TOOLCHAIN),flox)
FLOX_NODE_EXEC := $(FLOX_QUIET) activate --dir "$(CURDIR)/tools/nodejs" --
PNPM := $(FLOX_NODE_EXEC) pnpm
else
PNPM := $(FLOX_EXEC) pnpm
endif

.PHONY: verify-toolchain
verify-toolchain: ## Verify the selected Flox or system toolchain
ifeq ($(TOOLCHAIN),flox)
	@if [ "$(FLOX_ENV_PROJECT)" = "$(CURDIR)" ]; then \
		echo "Using the active Flox environment"; \
	elif ! command -v "$(FLOX)" >/dev/null 2>&1; then \
		echo "flox is required. Install it, then rerun this command."; \
		exit 1; \
	fi
else
	@bash script/verify-system-toolchain.sh
endif

.PHONY: verify-compiler
verify-compiler: ## Verify a C compiler is available for Python package builds
	@if ! command -v cc >/dev/null 2>&1; then \
		echo "A C compiler is required for Python package builds (for example, annoy)." >&2; \
		echo "Install Xcode Command Line Tools on macOS or build-essential on Debian/Ubuntu." >&2; \
		exit 1; \
	fi

.PHONY: verify-pnpm
verify-pnpm: verify-toolchain ## Verify pnpm is available for Studio bootstrap
ifeq ($(TOOLCHAIN),system)
	@bash script/verify-system-toolchain.sh web
endif
	@$(PNPM) --version || { \
		echo "pnpm not found."; \
		echo "Install Flox or use TOOLCHAIN=system with pnpm on PATH."; \
		exit 1; \
	}

.PHONY: verify-node-version
verify-node-version: verify-pnpm ## Verify pnpm and Node.js satisfy Studio's package engine
	@echo "~~~~~~"
	@echo "verifying Node.js version from web/package.json engines"
	@$(FLOX_EXEC) bash script/verify-node-version.sh

.PHONY: bootstrap-studio
bootstrap-studio: verify-node-version ## Install web dependencies and build Studio assets for FastAPI
	@echo "~~~~~~"
	@echo "installing Studio web dependencies and building FastAPI assets"
	cd web && CI=true $(PNPM) install --frozen-lockfile
	cd web && $(PNPM) --filter nemo-studio-ui build:fastapi

.PHONY: bootstrap-plugins
bootstrap-plugins: .venv ## Install editable plugin packages not covered by the root uv workspace
	@echo "~~~~~~"
	@echo "installing editable local plugin packages: $(BOOTSTRAP_LOCAL_PLUGIN_DIRS)"
	@editable_args=""; \
	for plugin in $(BOOTSTRAP_LOCAL_PLUGIN_DIRS); do \
		if [ ! -f "$$plugin/pyproject.toml" ]; then \
			echo "configured local plugin $$plugin is missing pyproject.toml"; \
			exit 1; \
		fi; \
		editable_args="$$editable_args -e $$plugin"; \
	done; \
	if [ -n "$$editable_args" ]; then \
		. .venv/bin/activate && $(UV) pip install $$editable_args; \
	else \
		echo "no local plugin packages configured, skipping"; \
	fi

.PHONY: bootstrap
bootstrap: bootstrap-python ## Bootstrap the local dev environment, including Studio assets
	@if ! $(MAKE) bootstrap-studio; then \
		echo ""; \
		echo "warning: optional Studio asset bootstrap did not complete."; \
		echo "Studio will be unavailable at http://localhost:8080/studio/ until assets are built."; \
		echo "Check the output above, then rerun:"; \
		echo "  make bootstrap-studio"; \
	fi
	@echo "bootstrap completed"
	@$(BOOTSTRAP_ACTIVATION_REMINDER)

.PHONY: run
run: build-policy ## Run the NeMo Platform locally with Docker job backend
	NMP_CONFIG_FILE_PATH=${NMP_CONFIG_FILE_PATH} $(UV) run nemo services run

.PHONY: clean
clean: clean-python ## Clean the NeMo Platform DB, files, and Python virtual environment
	rm -f /tmp/nmp-platform.db*
	rm -rf /tmp/nmp-files

.PHONY: update-licenses
update-licenses: ## Update the third_party/license.txt file with the latest licenses
	$(UV) sync --inexact
	$(UV) run --frozen nemo-platform-sdk-tools license generate

.PHONY: check-licenses
check-licenses: ## Check that license files are up to date
	LICENSE_DIR="$$(pwd)/third_party" && \
	export LICENSE_NAME="licenses_ci.jsonl" && \
	export PATH="$$HOME/.local/bin:$$PATH" && \
	$(MAKE) update-licenses && \
	diff third_party/licenses.jsonl "$${LICENSE_DIR}/$${LICENSE_NAME}" && \
	$(UV) run --frozen nemo-platform-sdk-tools license find-missing

CMD_COPYRIGHT_HEADER_FIXER := $(UV) run python tools/lint/copyright_fixer.py
.PHONY: update-copyright-headers
update-copyright-headers: ## Updates copyright headers across all files
	$(CMD_COPYRIGHT_HEADER_FIXER)

.PHONY: check-copyright-headers
check-copyright-headers: ## Checks to see if all copyright headers are appropriate
	$(CMD_COPYRIGHT_HEADER_FIXER) --check

.PHONY: lint
lint: ## Run all linters (licenses, openapi, config docs, python style/types/sdk, vendored SDK, CLI, auth config)
	$(FLOX_EXEC) bash tools/lint/lint-all.sh

LINT_FIX_VERIFY ?= 0

.PHONY: lint-fix
lint-fix: ## Auto-fix lint issues (set LINT_FIX_VERIFY=1 to also run CI lint checks)
	LINT_FIX_VERIFY=$(LINT_FIX_VERIFY) $(FLOX_EXEC) bash tools/lint/lint-fix.sh

.PHONY: test-github-scripts
test-github-scripts: verify-node-version ## Run unit tests for GitHub JavaScript scripts
	@test -x web/node_modules/.bin/eslint || { \
		echo "Studio dependencies are required. Run make bootstrap-studio first."; \
		exit 1; \
	}
	$(FLOX_NODE_EXEC) node --test .github/scripts/tests/*.test.cjs

.PHONY: lint-github-scripts
lint-github-scripts: verify-node-version ## Lint and format-check GitHub JavaScript scripts
	@test -x web/node_modules/.bin/eslint || { \
		echo "Studio dependencies are required. Run make bootstrap-studio first."; \
		exit 1; \
	}
	$(FLOX_NODE_EXEC) web/node_modules/.bin/eslint --config .github/eslint.config.mjs .github/scripts --max-warnings 0
	$(FLOX_NODE_EXEC) web/node_modules/.bin/prettier --check .github/scripts

.PHONY: check-github-scripts
check-github-scripts: test-github-scripts lint-github-scripts ## Test, lint, and format-check GitHub JavaScript scripts

.PHONY: vendor
vendor: ## Vendor packages into the SDK and generate wrapper metadata
	$(UV) run --no-sync nemo-platform-sdk-tools vendor all-from-configs \
		nemo_platform_ext models filesets \
		nemo_evaluator_sdk
	$(UV) run --no-sync nemo-platform-sdk-tools post-generation update-license-headers

# ============================================================================
# Python Testing Targets
# ============================================================================

.PHONY: test
test: test-unit  ## Run all Python unit tests (fast tests without infrastructure dependencies)


# In CI (e.g. GitLab sets CI=true): quiet progress for readable logs. Locally: verbose.
PYTEST_VERBOSITY := $(if $(filter true,$(CI)),-q,-v)
PYTEST_WORKERS ?= auto
PYTEST_MAX_WORKERS ?= 16
PYTEST_DIST ?= loadscope
PYTEST_MAX_WORKER_RESTART ?= 2
PYTEST_CMD = env PYTHONWARNINGS="ignore::UserWarning:pytest_only.version" $(UV) run --frozen \
	pytest \
	-n $(PYTEST_WORKERS) --maxprocesses=$(PYTEST_MAX_WORKERS) \
	--max-worker-restart=$(PYTEST_MAX_WORKER_RESTART) \
	--dist $(PYTEST_DIST) --timeout=120 $(PYTEST_VERBOSITY) $(PYTEST_EXTRA)

PYTEST_CI_OPTS := --cov=src --cov=packages \
	--junitxml=report.xml \
	--cov-report json:coverage.json \
	--cov-report xml:coverage.xml \
	--durations=25

# Global wall-clock timeout for CI test runs to prevent infinite hangs when
# a pytest-xdist worker crashes (e.g. SIGABRT from wasmtime) and doesn't exit.
# timeout sends SIGTERM after PYTEST_CI_TIMEOUT seconds, then SIGKILL after 60s.
PYTEST_CI_TIMEOUT ?= 1800
PYTEST_CI_CMD = timeout --kill-after=60s $(PYTEST_CI_TIMEOUT)s $(PYTEST_CMD)

# Unit/default runs keep ``loadscope`` so tests from the same module/class stay
# on one worker and reuse normal pytest fixtures efficiently. The integration
# targets collect tests that share real external resources (Docker containers,
# service ports, etc.). Those tests can mark the resource with ``xdist_group``,
# but pytest-xdist only honors those labels under ``loadgroup``. Both local and
# CI integration runs therefore use ``loadgroup`` while unit runs keep
# ``loadscope``.
test-integration test-integration-ci: PYTEST_DIST := loadgroup

# A killed integration worker never tears down its containers and ports, so replacing it re-runs
# the group against resources the dead worker still holds. Unit workers own nothing external.
test-integration test-integration-ci: PYTEST_MAX_WORKER_RESTART := 0

.PHONY: test-unit
test-unit: ## Run Python unit tests across all packages and services
	@echo "Running Python unit tests across all packages and services..."
	$(PYTEST_CMD) -m unit

.PHONY: test-unit-ci
test-unit-ci: ## Run Python unit tests with coverage for CI
	@echo "Running Python unit tests with coverage..."
	$(PYTEST_CI_CMD) $(PYTEST_CI_OPTS) -m unit

.PHONY: test-integration
test-integration: ## Run Python integration tests (tests service interfaces and interactions)
	@echo "Running Python integration tests..."
	$(PYTEST_CMD) -m integration

.PHONY: test-integration-ci
test-integration-ci: ## Run Python integration tests (tests service interfaces and interactions)
	@echo "Running Python integration tests..."
	$(PYTEST_CI_CMD) $(PYTEST_CI_OPTS) -m integration

.PHONY: test-gpu-integration
test-gpu-integration: ## Run Python gpu integration tests (tests service interfaces and interactions)
	# We run tests serially, because if any use the GPU they cannot share it
	@echo "Using extra: ${EXTRA}"
	@echo "Running Python gpu integration tests..."
	$(PYTEST_CMD) -m gpu_integration


.PHONY: test-gpu-integration-ci
test-gpu-integration-ci: ## Run Python integration gpu tests (tests service interfaces and interactions)
	# We run tests serially, because if any use the GPU they cannot share it
	@echo "Using extra: ${EXTRA}"
	@echo "Running Python gpu integration tests..."
	$(PYTEST_CI_CMD) $(PYTEST_CI_OPTS) -m gpu_integration

.PHONY: test-all-script
test-all-script: ## Run all unit tests using the helper script (with summary)
	@echo "Running all unit tests with summary..."
	$(UV) run --frozen python tools/run_all_tests.py

.PHONY: test-e2e
test-e2e: ## Run e2e tests against nemo services (starts/stops services automatically)
	@echo "Running e2e tests..."
	$(UV) run --frozen pytest e2e -v --run-e2e --junitxml=report.xml $(PYTEST_EXTRA)

.PHONY: test-regression
test-regression: ## Run Python regression tests (functional microservice baseline tests)
	@echo "Running Python regression tests..."
	$(UV) run --frozen pytest -v -m regression

.PHONY: test-all
test-all: ## Run all Python tests (unit, integration, e2e, regression)
	@echo "Running all Python tests..."
	$(UV) run --frozen pytest -v -m "not canary and not skip_in_ci"

.PHONY: test-canary
test-canary: ## Run canary tests against deployed environments
	@echo "Running canary tests..."
	$(UV) run --frozen pytest -v -m canary

.PHONY: test-coverage
test-coverage: ## Run tests with coverage reporting
	@echo "Running tests with coverage..."
	$(UV) run --frozen pytest -v --cov --cov-report=html --cov-report=term --cov-report=xml
	@echo "Coverage report generated in htmlcov/index.html"

.PHONY: test-coverage-report
test-coverage-report: ## Generate and display coverage report (run after test-coverage)
	@echo "Opening coverage report..."
	@command -v open >/dev/null 2>&1 && open htmlcov/index.html || \
	 command -v xdg-open >/dev/null 2>&1 && xdg-open htmlcov/index.html || \
	 echo "Coverage report is at htmlcov/index.html"

.PHONY: test-package
test-package: ## Run tests for a specific package (usage: make test-package PACKAGE=nmp_common)
ifndef PACKAGE
	$(error PACKAGE is not set. Usage: make test-package PACKAGE=<package_name>)
endif
	@echo "Running tests for package: $(PACKAGE)..."
	$(UV) run --frozen pytest -v packages/$(PACKAGE)/tests/

.PHONY: test-deployments-openshell
test-deployments-openshell: ## Run OpenShell deployment backend unit tests with the platform-restricted [openshell] extra installed
	# The openshell extra is not part of the default sync (platform-restricted
	# wheel), so the shared unit-test job skips these. Install it just here and
	# run the backend's tests for real (mirrors the nemo-guardrails --extra bench job).
	$(UV) run --frozen --package nemo-deployments-plugin --extra openshell \
		pytest -v plugins/nemo-deployments/tests/unit/backends/openshell/

.PHONY: test-service
test-service: ## Run tests for a specific service (usage: make test-service SERVICE=evaluator)
ifndef SERVICE
	$(error SERVICE is not set. Usage: make test-service SERVICE=<service_name>)
endif
	@echo "Running tests for service: $(SERVICE)..."
	$(UV) run --frozen pytest -v services/$(SERVICE)/tests/


.PHONY: test-fast
test-fast: ## Run fast tests only (excludes slow, e2e, integration marked tests)
	@echo "Running fast tests..."
	$(UV) run --frozen pytest -v -m "unit and not slow"

.PHONY: test-watch
test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "Running tests in watch mode..."
	$(UV) run --frozen ptw -- -v

.PHONY: test-debug
test-debug: ## Run tests with debugging output (verbose, no capture, show locals)
	@echo "Running tests in debug mode..."
	$(UV) run --frozen pytest -vvv -s --tb=long --showlocals

.PHONY: test-failed
test-failed: ## Re-run only the tests that failed in the last run
	@echo "Re-running failed tests..."
	$(UV) run --frozen pytest -v --lf

.PHONY: test-clean
test-clean: ## Clean test artifacts (coverage, cache, etc.)
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .tox
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Test artifacts cleaned"

.PHONY: test-list
test-list: ## List all available tests without running them
	@echo "Listing all tests..."
	$(UV) run --frozen pytest --collect-only -q

.PHONY: test-markers
test-markers: ## List all available pytest markers
	@echo "Available pytest markers:"
	$(UV) run --frozen pytest --markers

.PHONY: test-policy
test-policy: ## Run OPA policy tests for auth service
	@if ! command -v opa >/dev/null 2>&1; then \
		echo ""; \
		echo "ERROR: OPA (Open Policy Agent) is not installed."; \
		echo ""; \
		echo "For more info: https://www.openpolicyagent.org/docs/latest/#running-opa"; \
		exit 1; \
	fi
	@echo "Running OPA policy tests..."
	opa test services/core/auth/src/nmp/core/auth/app/policies services/core/auth/src/nmp/core/auth/app/policy_tests services/core/auth/src/nmp/core/auth/assets/static-authz.yaml -v

# Policy WASM bundle paths
ASSETS_DIR := services/core/auth/src/nmp/core/auth/assets

.PHONY: build-policy
build-policy: ## Build OPA policies into WASM bundle
	./script/build_policy_wasm.sh

.PHONY: check-policy
check-policy: ## Verify WASM bundle is up-to-date with policies
	@echo "Checking if policy.wasm is up-to-date..."
	@cp $(ASSETS_DIR)/policy.wasm $(ASSETS_DIR)/policy.wasm.bak 2>/dev/null || true
	@$(MAKE) build-policy >/dev/null 2>&1
	@if ! cmp -s $(ASSETS_DIR)/policy.wasm $(ASSETS_DIR)/policy.wasm.bak 2>/dev/null; then \
		mv $(ASSETS_DIR)/policy.wasm.bak $(ASSETS_DIR)/policy.wasm 2>/dev/null || true; \
		echo "ERROR: policy.wasm is out of date. Run 'make build-policy' and commit the result."; \
		exit 1; \
	fi
	@rm -f $(ASSETS_DIR)/policy.wasm.bak
	@echo "policy.wasm is up-to-date"


.PHONY: build-jobs-launcher
build-jobs-launcher:
	@echo "Building jobs launcher binary..."
	cd services/core/jobs/jobs-launcher/ && $(FLOX_EXEC) ./build-manual.sh

.PHONY: test-jobs-launcher
test-jobs-launcher:
	@echo "Testing jobs launcher binary..."
	cd services/core/jobs/jobs-launcher/ && $(FLOX_EXEC) go test ./... -v

.PHONY: test-e2e-docker
test-e2e-docker: ## Run e2e tests using docker
	@echo "Running e2e tests with docker..."
	$(UV) run --frozen pytest e2e --docker -v --junitxml=report.xml $(PYTEST_EXTRA)

.PHONY: test-e2e-docker-auth
test-e2e-docker-auth: ## Run e2e tests using docker and auth
	@echo "Running e2e tests with docker and auth..."
	$(UV) run --frozen pytest e2e --docker --feature auth -v --junitxml=report.xml

.PHONY: test-e2e-docker-gpu
test-e2e-docker-gpu: ## Run GPU e2e tests using docker (requires GPU host and GPU config)
	@echo "Running GPU e2e tests with docker..."
	$(UV) run --frozen pytest e2e --docker --feature gpu -v --junitxml=report.xml

.PHONY: test-e2e-kubernetes
test-e2e-kubernetes: ## Run e2e tests against Kubernetes (set NMP_E2E_CLUSTER_URL)
	@echo "Running e2e tests with Kubernetes..."
	$(UV) run --frozen pytest e2e --kubernetes -v -n 2 --junitxml=report-kubernetes.xml

.PHONY: test-e2e-kubernetes-network-policies
test-e2e-kubernetes-network-policies: ## Set up local kind with Calico and run the Chainsaw NetworkPolicy smoke test
	@echo "Running Chainsaw NetworkPolicy e2e smoke test..."
	e2e/k8s/scripts/run_network_policy_e2e.sh

.PHONY: test-e2e-kubernetes-auth
test-e2e-kubernetes-auth: ## Run e2e tests against Kubernetes with auth enabled (set NMP_E2E_CLUSTER_URL)
	@echo "Running e2e tests with Kubernetes and feature auth enabled..."
	$(UV) run --frozen pytest e2e --kubernetes --feature auth -n 2 -v --junitxml=report-kubernetes-auth.xml

.PHONY: test-e2e-kubernetes-kai
test-e2e-kubernetes-kai: ## Run KAI Scheduler e2e tests against Kubernetes (set NMP_E2E_CLUSTER_URL)
	@echo "Running e2e tests with Kubernetes and feature kai-scheduler..."
	$(UV) run --frozen pytest e2e --kubernetes --feature kai-scheduler -v --junitxml=report-kubernetes-kai.xml

.PHONY: test-e2e-kubernetes-gpu
test-e2e-kubernetes-gpu: ## Run GPU e2e tests against Kubernetes (requires GPU nodes; set NMP_E2E_CLUSTER_URL)
	@echo "Running GPU e2e tests with Kubernetes with feature gpu enabled..."
	$(UV) run --frozen pytest e2e --kubernetes --feature gpu -v --junitxml=report-kubernetes-gpu.xml

.PHONY: test-e2e-kubernetes-gpu-automodel
test-e2e-kubernetes-gpu-automodel: ## Run GPU automodel customization e2e tests against Kubernetes (requires GPU nodes; set NMP_E2E_CLUSTER_URL)
	@echo "Running GPU automodel customization e2e tests with Kubernetes..."
	$(UV) run --frozen pytest tests/agentic-use/customizer-lora-job-cli/tests/test_outputs.py --kubernetes --feature gpu --log-cli-level=INFO -v --junitxml=report-kubernetes-gpu-automodel.xml

.PHONY: benchmark-guardrails
benchmark-guardrails: ## Run nemo-guardrails IGW benchmark sweep (set BENCHMARK_ARGS for extra flags)
	@echo "Running nemo-guardrails IGW benchmark..."
	$(UV) run --frozen --package nemo-guardrails-plugin --extra bench \
		python -m nemo_guardrails_plugin.benchmarks.run $(BENCHMARK_ARGS)
