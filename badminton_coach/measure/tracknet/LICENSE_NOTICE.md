# Vendored from RacketVision

`model.py` in this directory is adapted (near-verbatim; see file header) from
[OrcustD/RacketVision](https://github.com/OrcustD/RacketVision) `source/BallTrack/model/tracknet_v3.py`
and `model/loss_utils.py` (AAAI 2026 — "RacketVision: A Multiple Racket Sports Benchmark for Unified
Ball and Racket Analysis"), released under the MIT License:

```
MIT License

Copyright (c) 2025 OrcustD

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The `balltrack_best.pth` checkpoint (downloaded separately into `data/models/racketvision/`, not
committed to this repo) is from the same project, released on Hugging Face
(`linfeng302/RacketVision-Models`).

The pre/postprocessing logic in `badminton_coach/measure/shuttle.py` (batch construction, heatmap
contour peak-finding, confidence calculation) is likewise adapted from RacketVision's
`source/BallTrack/inference.py`, restructured to run on in-memory video frames and an in-clip median
background rather than a pre-extracted frame directory + precomputed `median.npz`.
