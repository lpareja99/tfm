# Exploratory work

Earlier approaches explored during the thesis **before** converging on the final
method (parametrized multi-backbone Mask2Former under `experiments/`, see the root
`Readme.md`). Kept deliberately as a record of the research path — what was tried
and why it was replaced. **None of this is part of the final reproducible pipeline,
and none of it is maintained.**

| Folder | What it was | Why replaced |
|---|---|---|
| [`initial_mask2former/`](initial_mask2former/README.md) | First **monolithic** implementation: all backbones + Azure jobs + CVAT data-prep + analysis in one project. | Split into self-contained per-backbone experiments under `experiments/`. |
| [`hugging_faces_trial/`](hugging_faces_trial/README.md) | Training via the **HuggingFace Transformers** stack (crack classes only). | MMSegmentation offered the swappable-backbone + config-inheritance + `mim` tooling the thesis needed. |

See each folder's `README.md` for details (structure, how it was run, caveats).

> The full original history (including unscrubbed Azure/ACR identifiers) is
> preserved in the pre-cleanup backup branches/tags.
