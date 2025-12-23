# Changelog

All notable changes to BabelVec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
