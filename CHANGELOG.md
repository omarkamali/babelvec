# Changelog

All notable changes to BabelVec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-01-03

### Added
- **Thread Control**: Exposed `threads` parameter in `train_monolingual` and `train_multilingual` to manually specify FastText thread count.

### Changed
- Renamed `TrainingConfig.thread` to `TrainingConfig.threads` for consistency with other parameters.

## [0.1.5] - 2025-12-27

### Added
- **Language Family Assignment System**: 100 hardcoded families covering all 355 Wikipedia languages
- Family grouping API: `get_family_key()`, `get_family_languages()`, `assign_families()`, `get_training_groups()`
- Hybrid training strategy support for joint vs separate model training

## [0.1.4] - 2024-12-23

### Added
- **Parallel Training**: New `train_multiple_languages()` function for training multiple languages simultaneously
- **FastText Parameters**: Exposed `loss` and `bucket` parameters for fine-tuning

### Changed
- `TrainingConfig.thread` now defaults to auto-detected CPU count instead of hardcoded 4
- Training now prints thread count for visibility

## [0.1.3] - 2024-12-23

### Fixed
- **Critical Bug**: Projection matrices are now properly saved and loaded with `.bin` files
  - Previously, saving aligned models as `.bin` would lose the projection matrix
  - Now saves companion `.projection.npy` and `.meta.json` files alongside `.bin`
  - Loading `.bin` files automatically loads projection if available

### Changed
- Model save format for `.bin` files now includes:
  - `{name}.bin` - FastText model
  - `{name}.projection.npy` - Projection matrix (if aligned)
  - `{name}.meta.json` - Metadata (language, dimension, etc.)

## [0.1.2] - 2024-12-22

### Added
- Initial release with core functionality
- FastText-based word embeddings
- Position-aware sentence encoding (RoPE, Sinusoidal, Decay)
- Cross-lingual alignment (Procrustes, InfoNCE, Ensemble)
- Evaluation utilities (retrieval, similarity, coverage)

## [0.1.1] - 2024-12-21

### Added
- Basic project structure
- Core model implementation
- Training utilities

## [0.1.0] - 2024-12-20

### Added
- Initial project setup
