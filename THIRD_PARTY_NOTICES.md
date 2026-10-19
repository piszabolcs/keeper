# Third-party notices

Keeper's own code is MIT licensed (see `LICENSE`). It builds on the
components below. Only two of them are **redistributed in this repository**
(the SCUNet network definition and the YuNet face model); the libraries are
installed by pip from their official packages and keep their own licences.

## Redistributed in this repository

### SCUNet network definition – `vendor/scunet.py`

- Source: https://github.com/cszn/SCUNet (`network_scunet.py`), Kai Zhang et al.
- Licence: Apache License 2.0 – full text in `licenses/Apache-2.0.txt`.
- Modified: the `timm` and `thop` imports and the per-block `print` calls were
  removed; the changes are listed in the file header.
- The pretrained weights (`scunet_color_real_psnr.pth`, from the KAIR release
  of the same authors, Apache-2.0) are **not** in the repository;
  `tools/download_models.py` downloads them and checks their SHA-256.
- Citation: Zhang et al., "Practical Blind Image Denoising via Swin-Conv-UNet
  and Data Synthesis", Machine Intelligence Research, 2023.

### YuNet face detection model – `assets/face_detection_yunet.onnx`

- Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  (YuNet by Shiqi Yu et al., from the libfacedetection project).
- Licence: MIT – the copyright and permission notice are in the `LICENSE`
  file next to the model in the opencv_zoo repository and apply to this copy.

## Installed dependencies (not redistributed)

| Component | Licence | Note |
|---|---|---|
| [PySide6 / Qt for Python](https://doc.qt.io/qtforpython/) | LGPL-3.0 (or commercial) | dynamically linked through the pip wheels |
| [OpenCV (opencv-python)](https://github.com/opencv/opencv-python) | Apache-2.0 | the Windows wheel also ships FFmpeg (LGPL-2.1) |
| [NumPy](https://numpy.org/) | BSD-3-Clause | |
| [Pillow](https://python-pillow.org/) | MIT-CMU (HPND) | |
| [rawpy](https://github.com/letmaik/rawpy) | MIT | |
| [LibRaw](https://www.libraw.org/) (inside rawpy) | LGPL-2.1 or CDDL-1.0 | the rawpy wheel ships it under LGPL-2.1, text in `licenses/LGPL-2.1.txt` |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | |
| [PyTorch](https://pytorch.org/) – optional | BSD-3-Clause | learned denoiser only |
| [einops](https://github.com/arogozhnikov/einops) – optional | MIT | learned denoiser only |
| [pytest](https://pytest.org/), [pytest-qt](https://github.com/pytest-dev/pytest-qt) – development | MIT | tests only |

"rawpy MIT" alone would be misleading: rawpy's own code is MIT, but the RAW
decoding is done by the bundled LibRaw library, which is LGPL-2.1 / CDDL-1.0.
