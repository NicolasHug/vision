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
