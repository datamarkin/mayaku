# `tools/` — maintainer scripts

Scripts in this directory are *not* part of the installed `mayaku`
package. They support release engineering (weight hosting) and are not
documented in user-facing docs.

- `generate_manifest.py` — walk `models/` and write the `manifest.json` the
  weight downloader reads (path, size, sha256 per model revision).
- `zip_mlpackages.py` — zip every `.mlpackage` directory under `models/` so it
  can be hosted as a single file.
