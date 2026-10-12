# Research anonymization tools

The random, pool, and GAN embedding anonymizers are preserved for research use.
They are separate from the supported voice-conversion API. Their outputs can be
supplied to inference when they match the expected x-vector-plus-ECAPA `[1, 704]`
conditioning representation; validating privacy and quality remains the caller's
research task.

Model weights, generated embedding pools, and used-index files under `models/`
are local artifacts and are no longer tracked. Existing local copies were
preserved. Pool/GAN experiments require their original artifacts and settings;
they are not included in the first inference model bundle. Run research modules
from the repository root after installing training dependencies.
