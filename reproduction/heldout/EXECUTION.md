# Execution record

The seven-group internal test was evaluated with all six previously published checkpoints. The three fixed epoch-99
models form the primary comparison; the three earlier checkpoints remain secondary because external sets helped
select them. [RESULTS.md](RESULTS.md) reports the measurements and [execution_protocol.json](execution_protocol.json)
records timestamps, software, hashes, model identities, thresholds and locked parameters.

## Inputs and inference

The frozen index contains 6,592 native frames (3,296 per teacher-derived class) from seven groups and 4,000 deterministic
synthetic recipes from 2,000 paired originals. No newly processed frames or new teacher labels were added. Declared
video-group identities and photo-original paths are disjoint across training, validation and test. This identity check
does not exclude unknown near-duplicates.

Each source file and mask was hashed privately before preparation. The scorer checked each prepared cache checksum
before inference. Native frames were decoded, cropped and deterministically masked without additional video encoding;
photo variants were generated once with the pinned current synthesiser and reused across all models. The original
configuration's native and synthetic mask shares were both 0.5. All shape-bucket tails were retained.

The pinned synthesiser includes encoder retries and the later ID3-byte workaround described in METHOD §11. Its hashes
identify the version actually evaluated; this does not assert byte-identical historical synthesis or unchanged pixels
throughout training. Public protocol hashes summarise inventories without releasing paths or per-image records.

Inference used an RTX 5090, CUDA bfloat16 autocast, batch size 8 and a process memory fraction of 0.70. There was no
test-time augmentation. A private copy of the trainer's model implements the historical autocast path, loaded with the
published weight dictionaries. Before scoring test images, the public and trainer implementations gave exactly the
same scores on eight existing openly licensed demos for each of the three epoch-99 input modes. Synthetic zero arrays
at the largest cached geometry provided the memory preflight. All six published dictionaries were also checked
tensor-for-tensor against their corresponding training checkpoints.

Each checkpoint retained its published r90/r95/r98 validation thresholds. These match quantiles of the historical
six-decimal validation-score exports. Test probabilities were saved with 17 significant digits; thresholds were never
refitted on test scores or on bootstrap samples. Analysis requires six complete score sidecars bound to the unchanged
locked manifest, checkpoint hashes, exact cohort and actual CSV checksums.

## Access history and source versions

An earlier evaluator had already read test metadata and membership of the measurement-agreement subset. It stalled
while spawning Windows data-loader workers and produced no saved predictions. Its process was stopped after that
condition was diagnosed. These records do not establish that nobody ever used the test historically, and the protocol
is not retroactive preregistration.

The user stopped the competing GPU workload before this execution. The replacement runner has a Windows main guard,
threaded cache loading and a private output directory. It completed one locked prediction set of 63,552 scores; no new
training, checkpoint search or teacher labelling was performed.

[executed_runner_v1.py](executed_runner_v1.py) preserves the byte-exact source whose hash is in the executed manifest.
The maintained [run_test.py](run_test.py) subsequently adds checks for completed-cache integrity, recovery of a fully
written CSV interrupted before its completion marker, and explicit selection-file binding for future preparation.
These changes were tested on CPU after scoring; the completed experiment was not rerun. The v1 runner checked the
frozen index but did not itself check the separate selection file. That selection SHA-256 was independently rechecked
against the preserved study record during execution.

## Analysis and interpretation

The analysis uses 2,000 paired whole-video-group bootstrap draws and 2,000 paired-original photo draws, seed 20261007.
It reports pooled metrics, group-specific counts and rates, equal-group mean operating points, seven leave-one-group-out
analyses, and paired primary-model differences. The largest group supplies 63.8% of native frames, so the pooled result
does not describe seven equally weighted sources. Seven clusters provide limited information about population
uncertainty; intervals condition on one trained model per input, fixed thresholds and the retained labels.

Frame metrics describe agreement with filtered teacher labels; synthetic metrics describe the construction labels.
Neither supplies independent human accuracy. Raw scores, original images, masks, sample identifiers, private paths,
private manifests and the trainer modules remain non-public. The aggregate analysis and execution sources are
published, while independent image-level reproduction still requires the private inputs and trainer dependencies.
