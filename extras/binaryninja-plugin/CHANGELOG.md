# Changelog

All notable changes to the Undertale Binary Ninja plugin are documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Undertale > Rate Function Name** command to rate a generated function name
  (Bad, Fair, Average, Good, Great) with optional comments. Feedback is sent to
  the inference server that generated the name, and the form is prefilled with
  any rating already stored there. Rating is disabled for functions whose name
  changed after Undertale named them, whether renamed by the user or because
  the rename was undone.

### Changed

- A rejected login token is now detected from both `401` and `422` responses,
  so missing or expired tokens prompt for credentials again.

## [1.0.0-beta.0] - 2026-09-30

### Added

- Initial release.
- **Undertale > Generate Function Name** command that sends a function's
  disassembly to an Undertale Inference Server and renames the function to the
  predicted name.
- Configurable inference server connection (TCP `host:port` or Unix domain
  socket) and base URL.

[Unreleased]: https://github.com/undertale-re/undertale/compare/binaryninja-plugin/v1.0.0-beta.0...HEAD
[1.0.0-beta.0]: https://github.com/undertale-re/undertale/tree/binaryninja-plugin/v1.0.0-beta.0
