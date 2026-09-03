# scratch/

Throwaway investigation, kept because the findings shaped the design.

- **`spike_topozarr.py`** — the pre-implementation spike. Establishes that
  driving `topozarr.engine`'s kernel directly gives a fill-aware mean that
  matches an independent reference bit-for-bit, and *reproduces the trap* it
  avoids: with `fill_value=None`, a coastal cell whose correct value is 352 comes
  back as **−16207**. Also pins down `boundary="trim"` level shapes and the
  GeoZarr attrs for a CRS with no EPSG code. Run it with
  `uv run python scratch/spike_topozarr.py`.

`usda_cropland_data/scratch/probe_source_coop.py` is deliberately **not** ported.
It exists to answer one question -- does Source Coop's S3 API support enough for
icechunk to commit directly, or is a local-build-then-sync needed? -- and that
question is already settled: the playbook's own §9 is "commit straight into the
remote store" with sync only as a fallback, `usda_gnatsgo` commits directly to
`data.source.coop` today, and the one probe finding that still matters (no batch
`DeleteObjects`, path-style addressing only) is encoded in the `remote.py` this
repo ports verbatim.

The real first test against the remote is `make init-store ACCOUNT=chill`: one
commit, exercising the exact path that matters (conditional PUT on the pointer
file, path-style addressing, credential refresh), and reversible with
`make clean-remote-store`. If Source Coop has regressed, that fails immediately
and icechunk says why.
