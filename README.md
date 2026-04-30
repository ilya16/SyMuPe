# SyMuPe: Affective and Controllable Symbolic Music Performance

> Official release for the paper [**"SyMuPe: Affective and Controllable Symbolic Music Performance"**](https://dl.acm.org/doi/10.1145/3746027.3755871)
> (**ACM MM 2025 Outstanding Paper Award**)
> 
> Proceedings of the [33rd ACM International Conference on Multimedia (ACM MM '25), Dublin, Ireland, 2025](https://acmmm2025.org/)
>
> Authors: Ilya Borovik, Dmitrii Gavrilev, and Vladimir Viro
>
> [![arXiv](https://img.shields.io/badge/arXiv-2511.03425-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.03425)
> [![ACM DL](https://img.shields.io/badge/MM_'25-Proceedings-19552e?logo=acm&logoColor=white)](https://dl.acm.org/doi/10.1145/3746027.3755871)
> [![Outstanding Paper Award](https://img.shields.io/badge/MM_'25-Outstanding_Paper-E6712D.svg)](https://acmmm2025.org/awards/)
> [![Website](https://img.shields.io/badge/Website-Demo-2563eb)](https://ilya16.github.io/SyMuPe)
> [![HF](https://img.shields.io/badge/HuggingFace-Models_&_Data-yellow?logo=huggingface)](https://huggingface.co/SyMuPe)

## Description

**SyMuPe** is a framework for creating controllable, transformer-based models for rendering symbolic music performances.

Its flagship model, **PianoFlow**, applies conditional flow matching to solve diverse multi-mask performance inpainting tasks.
By design, the model supports both **unconditional generation** and **infilling** of expressive performance features.

For more details, please refer to the [paper](https://arxiv.org/abs/2511.03425) and the [demo page](https://ilya16.github.io/SyMuPe) with samples.

## Install

Install `symupe` package using:
```shell
pip install -U symupe
```

## Models

Starting with v1.1.0, SyMuPe supports a *unified inference API* for the trained symbolic music models.

All models are grouped into three categories:
1. **Generators** (e.g. `PerformanceGenerator`)
2. **Classifiers** (e.g. `MusicClassifier`)
3. **Embedders** (e.g. `MusicEmbedder`)

The models can be loaded using the corresponding `AutoFactory` classes (`AutoGenerator`, `AutoClassifier`, or `AutoEmbedder`).

The trained models are available and documented on the [Hugging Face Hub](https://huggingface.co/SyMuPe).

### Score-to-Performance Rendering

<img alt="PianoFlow architecture" src="https://raw.githubusercontent.com/ilya16/SyMuPe/main/assets/pianoflow.png">

Score-only performance rendering models described in the paper are listed in the [SyMuPe (ACM MM'25) Collection](https://huggingface.co/collections/SyMuPe/symupe-acm-mm25) on Hugging Face.

| Model Repo | Type                       | Objective | Description                                 |
|:---|:---------------------------|:---|:--------------------------------------------|
| [**PianoFlow-base**](https://huggingface.co/SyMuPe/PianoFlow-base) | Encoder Transformer | CFM | Flagship model for high-fidelity rendering  |
| [**EncDec-base**](https://huggingface.co/SyMuPe/EncDec-base) | Encoder-Decoder Transformer | CLM | Slower sequence-to-sequence baseline        |
| [**MLM-base**](https://huggingface.co/SyMuPe/MLM-base) | Encoder Transformer | MLM | Fast single-step language modeling baseline |

#### Quick Start

Render an expressive performance from a quantized MIDI score using an example code snippet:

```python
import torch
from symupe import AutoGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select model name from the Hub
model_name = "SyMuPe/PianoFlow-base"
# model_name = "SyMuPe/EncDec-base"
# model_name = "SyMuPe/MLM-base"

# Build Generator by loading the model and tokenizer directly from the Hub
generator = AutoGenerator.from_pretrained(model_name, device=device)

# Perform score MIDI (tokenization is handled inside)
gen_results = generator.perform_score(
    "score.mid",
    use_score_context=True,
    num_samples=8,
    seed=23,
)
# gen_results[i] is PerformanceRenderingResult(...) containing:
# - score_midi, score_seq, gen_seq, perf_seq, perf_midi, perf_midi_sus

# Save performed MIDI files in a single directory
generator.save_performances(gen_results, out_dir="samples")
```

### MIDI Quality Classification

The [MIDI Quality Classifier](https://huggingface.co/SyMuPe/MIDI-Quality-Classifier) can be used to classify a MIDI file 
into one of the four quality classes: `score`, `high quality`, `low quality`, or `corrupted`.

The classifier was presented in the article [**"PianoCoRe: Combined and Refined Piano MIDI Dataset."**](https://doi.org/10.5334/tismir.333)

#### Quick Start

```python
import torch
from symupe import AutoClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build Classifier by loading the model and tokenizer directly from the Hub
classifier = AutoClassifier.from_pretrained(
    "SyMuPe/MIDI-Quality-Classifier", device=device,
)
# model, tokenizer, labels = classifier.model, classifier.tokenizer, classifier.labels

# Classify MIDI (tokenization is handled inside)
result = classifier("performance.mid")
# result is MusicClassificationResult(...) containing:
# - midi, seq, probabilities, prediction, label, all_logits, all_probabilities, all_predictions,
#   sequences and window_indices
print(result.label, result.probabilities)
```

## Datasets and Data Processing

The SyMuPe project provides two MIDI datasets for analysing and modeling piano expression:
1. The [**PERiScoPe**](https://huggingface.co/datasets/SyMuPe/PERiScoPe) **(Piano Expression Refined Score and Performance MIDI)** dataset, used to train the models in the [SyMuPe paper](https://arxiv.org/abs/2511.03425).
2. The [**PianoCoRe**](https://github.com/ilya16/PianoCoRe) **(Combined and Refined Piano MIDI)** dataset, presented in the eponymous article published in the [TISMIR journal](https://doi.org/10.5334/tismir.333).

### Refined Alignment for Scores and Performances

The refined subset of PianoCoRe was constructed using the **Refined Alignment for Scores and Performances (RAScoP)** pipeline,
integrated into the SyMuPe package in v1.1.0. The algorithm is described in the [article](https://doi.org/10.5334/tismir.333).

#### Quick Start

Use the following quick start code to align the score and performance MIDI:
```python
from symupe.data.alignments import ParangonarAligner, RAScoPConfig, RAScoP

score_midi_path = "score.mid"
perf_midi_path = "performance.mid"

# Align performance to score
aligner = ParangonarAligner()
alignment, paths, msgs = aligner.align(score_midi_path, perf_midi_path)

# Initialize RAScoP
config = RAScoPConfig(
    score_holes=True,
    performance_holes=True,
    clean_onsets=True,
    interpolate_notes=True,
    synchronize_performance=False,
    min_recall=0.7,
    num_runs=1,
)
rascop = RAScoP(
    score_midi=score_midi_path,
    perf_midi=perf_midi_path,
    alignment=alignment,
    config=config,
    verbose=1,
)

# Refine raw alignment
alignment, match_ratios, stage_times = rascop()
perf_midi, score_midi = rascop.perf_midi, rascop.score_midi  # refined MIDI
print(match_ratios)
```

Additional usage examples are available in the [PianoCoRe repository](https://github.com/ilya16/PianoCoRe).

The RAScoP pipeline expects both the score and the performance MIDI files to be single-track. 
During initialization, all tracks are merged and duplicate notes are removed by default. 
For any other preprocessing, use `symupe.data.midi.preprocess_midi`:
```python
from symusic import Score
from symupe.data.midi import preprocess_midi

midi = Score("score.mid")
midi = preprocess_midi(
    midi,
    to_single_track=True,
    clean_duplicates=True,
    cut_overlapped_notes=True,
    clean_short_notes=True,
    min_tick_shift=10,
    min_tick_duration=5,
    target_ticks_per_quarter=480,
)
midi.dump_midi("score_processed.mid")
```

### MusicXML to MIDI conversion

SyMuPe provides utilities for working with MusicXML files. The code extends [partitura](https://github.com/CPJKU/partitura)
by supporting the processing of duplicate and overlapping notes, expansion of grace notes, and deduplication of ornament notes, 
as described in the [PianoCoRe article](https://doi.org/10.5334/tismir.333).

Use `symupe.data.partitura.partitura_score_to_midi`, to convert score files to MIDI similar to the PERiScoPe/PianoCoRe datasets:
```python
import partitura as pt
from symupe.data.partitura import partitura_score_to_midi

score = pt.load_score("score.musicxml")
midi = partitura_score_to_midi(
    score,
    expand_grace_notes=True,
    process_ornaments=True,
    clean_duplicates=True,
    cut_overlapped_notes=True,
    downsample_ticks_per_quarter=48,
    ticks_per_quarter=480,
)
```

## Citation

If you use the package, models or the PERiScoPe dataset in your research, please cite:
```bibtex
@inproceedings{borovik2025symupe,
  title = {{SyMuPe: Affective and Controllable Symbolic Music Performance}},
  author = {Borovik, Ilya and Gavrilev, Dmitrii and Viro, Vladimir},
  year = {2025},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {10699--10708},
  doi = {10.1145/3746027.3755871}
}
```

If you use the PianoCoRe dataset or the RAScoP pipeline for data processing, please cite:
```bibtex
@article{borovik2026pianocore,
  title = {{PianoCoRe: Combined and Refined Piano MIDI Dataset}},
  author = {Borovik, Ilya},
  year = {2026},
  journal = {Transactions of the International Society for Music Information Retrieval},
  volume = {9},
  number = {1},
  pages = {144--163},
  doi = {10.5334/tismir.333}
}
```

## License

- The **source code** in this repository is licensed under the [Apache License 2.0](LICENSE).
- The **trained model weights** and the **datasets** are licensed under 
  [CC BY-NC-SA 4.0](LICENSE-DATA) license.
