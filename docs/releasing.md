# Releasing

The whole repository has one version ([Semantic Versioning](https://semver.org)):
every `pyproject.toml`, the matching `uv.lock` entries, `ntv3_mcp.__version__` and
`CITATION.cff` carry the same number, and each MCP server reports it as its
`serverInfo.version` (the gateway at `GET /version`). `make check-version` verifies
this, and CI runs it on every push.

- **patch** (0.9.1): bug fixes, docs, dependency bumps that don't change behaviour;
- **minor** (0.10.0, 1.1.0): new services, checkpoints, tools or optional parameters;
- **major** (2.0.0): a breaking change to tool names, parameters, response shapes, the
  gateway API or the compose setup. Before 1.0.0, minor releases may also break things.

## Steps

1. Make sure `main` is green (CI: lint, version check, tests of every project).
2. Set the version everywhere:
   ```bash
   make bump VERSION=0.9.1
   ```
3. In `CHANGELOG.md`, move the `[Unreleased]` items under a new
   `## [0.9.1] - YYYY-MM-DD` heading and update the compare links at the bottom.
4. Commit and tag:
   ```bash
   git commit -am "Release 0.9.1"
   git tag -a v0.9.1 -m "dnalm-hub 0.9.1"
   git push origin main v0.9.1
   ```
5. The `Release` workflow checks that the tag matches the version, then creates the
   GitHub release with that version's CHANGELOG section as release notes. Versions with
   a suffix (`1.0.0-rc.1`) become pre-releases.

## DOI (Zenodo)

To get a citable DOI for each release, enable the repository once at
<https://zenodo.org/account/settings/github/> (needs a Zenodo account linked to GitHub
with access to `kbi-fbmi`). From then on every GitHub release is archived and gets a
DOI; Zenodo reads the metadata from `CITATION.cff`. Add the concept DOI badge to the
README after the first archived release.

## Docker images

Images are built locally from `compose.yaml` (`make build`) and are not published to a
registry. Rebuild after upgrading so the services report the new version.
