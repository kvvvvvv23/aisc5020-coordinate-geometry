# Team release: part4-integrated-v1

Module 4: diagonal preconditioning and paired statistical analysis (HUANG Ze).

Contains P5 derivation, standalone numerical validation, a read-only analysis of
10,200 saved evaluation runs / 6,600 pairs, joint/cluster bootstrap (2,000 draws),
CSV tables, plots, module report PDF/LaTeX, tests and reproduction instructions.

Repository: [kvvvvvv23/aisc5020-coordinate-geometry](https://github.com/kvvvvvv23/aisc5020-coordinate-geometry).
The [team integration release](https://github.com/kvvvvvv23/aisc5020-coordinate-geometry/releases/tag/part4-integrated-v1)
uses tag `part4-integrated-v1` to identify the reviewed commit on `main`.
Release validation metadata records publication and asset checks.

## Contribution and data provenance

The original Module 4 contribution is by HUANG Ze, imported from
[acbldppx/aisc5020-part4-huangze](https://github.com/acbldppx/aisc5020-part4-huangze)
at commit `1d9b405e0626e2c9b24a2fcd165025f2e68b97df`.
The team imported a privacy-sanitized file snapshot and applied review fixes;
it did not merge the contributor's Git history. Authorship is retained.

The historical Part 3 execution is `575ebf4`. The original contribution used
team source baseline `bb365b9`; the new team integration has its own commit
identity. Statistical analysis of saved results is distinct from rerunning
the long-horizon main experiment. Historical input bytes and their source
hashes are retained separately from regenerated team reports and summaries.
The pins in `requirements-part4.txt` describe the contributor's analysis
environment; the team integration records its actual environment separately.

Important limitation: the contributor's recorded structural checks pass, but
its numerical replay does not fully pass: 2 eigenbasis-Adam runs among 30
sampled reruns exceed tolerance on 9 fields. The earlier team audit separately
found 56 mismatched runs among 450 samples. These are different checks, not
counts of divergent evaluation runs. New tests and deterministic statistical
recomputation do not resolve these historical optimizer replay differences.
The report claims neither fully reproduced long-horizon trajectories nor
controlled timing.

## Building and publishing the team version

1. Finish review fixes and the full test/default-reproduction checks on the
   team integration. Record the actual commands, environment and outcomes;
   do not relabel historical execution metadata with the current source.
2. Rebuild the report from the team's cleaned TeX, regenerate tables and
   figures from the identified historical inputs, and assemble a new ZIP.
   Check the new source, PDF, metadata and archive contents for personal
   identifiers and local credentials before publication.
3. Merge the reviewed integration into team `main`, tag that commit
   `part4-integrated-v1`, and attach the newly built PDF and clean ZIP.
   Verify downloadable assets and their hashes, then share the team links.

This cleanup applies to the team's imported snapshot and new release assets.
It does not modify or certify the contributor's existing repository history,
Release, ZIP or PDF. Those upstream assets must not be presented as the
team's cleaned release.

Large raw results belong in release assets rather than ordinary Git blobs.
See [the integration guide](part4_guide.md) for input paths and reproduction
commands. This module report remains a contribution to the final group report.
