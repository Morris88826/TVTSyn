# Example audio

The two mono 16 kHz files are unmodified recordings from the CMU ARCTIC
speech databases, created by John Kominek and Alan W. Black at Carnegie Mellon
University. They were retrieved from the publisher and verified byte for byte;
URLs and SHA-256 checksums are recorded in [provenance.json](provenance.json).

| Local file | Original recording |
| --- | --- |
| `source.wav` | `cmu_us_bdl_arctic/wav/arctic_a0001.wav` |
| `target.wav` | `cmu_us_slt_arctic/wav/arctic_a0001.wav` |

The original [COPYING](COPYING) notice applies to these recordings, separately
from the code license and TVTSyn model-weight license. Source:
[CMU ARCTIC](http://festvox.org/cmu_arctic/).

During release preparation, the historical BDL copy was found to differ from
the original recording. It was replaced by the verified original; the previous
local copies remain under `artifacts/samples_before_provenance_check/`, and the
historical `sample_wav/` directory remains untouched and untracked.
