# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning: [SemVer](https://semver.org/).

## [1.0.0] - 2026-10-19

### Added
- PySide6 desktop app that culls a folder of JPEG and Sony ARW photos by sharpness, with a 0 to 100 score
  from three focus metrics, noise compensation and EXIF-based adjustment for the shooting conditions.
- Series and duplicate detection (capture time and perceptual hash), best-of-series star, "keep best N".
- Face sharpness with the bundled YuNet model, used to catch background focus and refine the series pick.
- Live threshold, manual keep or reject overrides, header hints ("likely shaken", "likely noisy").
- Safe moves into `rejected/` and `duplicates/` (same-volume renames, never overwrite, RAW+JPEG pairs move
  together) with a write-ahead undo log and validated undo.
- Corrected copies into `corrected/` (denoising, sharpening, brightness, white balance, colour and local
  contrast) with a 1:1 preview; optional learned SCUNet denoiser on NVIDIA GPUs.
- Quick check for a single photo, `Keeper.bat` launcher with folder and photo arguments, rotating
  application log, About card, offscreen pytest suite and Windows CI.
