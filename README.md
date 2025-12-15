# SyMuPe: Affective and Controllable Symbolic Music Performance

<img alt="PianoFlow architecture" src="assets/pianoflow.png">

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
> [![Website](https://img.shields.io/badge/Demo-Samples-2563eb)](https://ilya16.github.io/SyMuPe)
> [![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey)](LICENSE)

## Description

**SyMuPe** is a framework for creating controllable, transformer-based models for rendering symbolic music performances.

Its flagship model, **PianoFlow**, applies conditional flow matching to solve diverse multi-mask performance inpainting tasks.
By design, the model supports both **unconditional generation** and **infilling** of expressive performance features.

For more details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3746027.3755871) and the [demo page](https://ilya16.github.io/SyMuPe) with samples.

## Citation

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

## License

The materials in this repository are licensed under the
[Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](LICENSE).
