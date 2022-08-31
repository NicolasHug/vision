- Philip:
  - Add support for data-loaders. For DPs we mostly care about DataLoader2 (https://github.com/pytorch/vision/pull/6196/files#diff-32b42103e815b96c670a0b5f0db055fe63f10fc8776ccbb6aa9b61a6940abba0R201-R209)
  - Add support for num_workers > 1 -- See note in FFCV's Loader about need for batch-size > 1

- Add support for WebDataset

- Add support for reading files from a thread pool.

- Run similar benchmarks internally - torchdata will dedicate time for this

- investigate why tar reading is slower than the rest of the archives. Vitaly
  is on it.
  - If tar ends up being faster, worth considering storing tensors or bytesio
    in the tar files as well?

- benchmark on BIG dataset, a lot bigger than ImageNet. Map-Style may struggle
  with these because it requires shuffling a huge array of indices

- Eventually: It'd be interesting to run FFCV with its built-in tranforms, but
  overriding the memory allocation with None as done here
  https://github.com/libffcv/ffcv/blob/f25386557e213711cc8601833add36ff966b80b2/ffcv/transforms/module.py#L30-L31
  This would give an idea of how much this memory allocation thing accounts
  for. Should be doable with a simple wrapper class. -->
