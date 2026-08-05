# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-28

### Added
- DTW, nDTW, and SDTW trajectory metrics
- PyPI badge and pip/uv install instructions to README
- Citation information and BibTeX entry
- Math extensions for Sphinx documentation
- Codecov action to CI pipeline
- `/release-frame` skill for automated semver releases

### Fixed
- Jekyll Liquid syntax error breaking GitHub Pages build
- Equation rendering in Sphinx documentation

### Changed
- Updated bibliography references

## [0.1.0] - 2025-01-01

Initial release of FRAME: Framework for Robotic Action and Motion Evaluation.

### Added
- `SuccessRate` task performance metric
- `TaskCompletionRate` metric
- `ActionAccuracy` metric (MSE, AMSE, NAMSE)
- `PathLength`, `PathSmoothness`, `CurvatureChange` trajectory quality metrics
- `AbsoluteTrajectoryError` and `RelativeTrajectoryError`
- `CollisionRate`, `ObstacleProximity`, `RiskFactor` safety metrics
- `InferenceLatency` and `MemoryUsage` efficiency metrics
- TorchMetrics-based distributed training support
- PyTorch Lightning and Hugging Face Transformers integration examples
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-29

### Changed
- **BREAKING**: `PathSmoothness` normalization changed from `1/PL` (path length)
  to `1/(L-2)` (number of direction-change samples). This decouples the smoothness
  metric from trajectory length, addressing a mathematical confound identified in
  peer review (ICML 2026 workshop). Values are not comparable with prior versions.
  Closes #15.
