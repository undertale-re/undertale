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
- **HTTPS** connections. The server's certificate and hostname are always
  verified, against a configurable CA certificate file (for self-signed
  certificates or a private certificate authority), the OS trust store when
  the optional `truststore` package is installed, or Python's default trust
  store. Certificate, TLS, and redirect errors are reported with hints on how
  to fix them.
- Before a login token or credentials are sent over plain HTTP to a host
  other than this machine, the plugin asks for confirmation, once per server
  per session. Nothing is sent unless the user agrees.

### Changed

- The TCP connection type is split into HTTP and HTTPS, and **Undertale >
  Configure Plugin** asks for the connection in two steps (type, then only
  that type's fields), like the first-use prompt.
- The base URL is part of the connection instead of a separate setting, so
  changing it counts as switching servers: it clears the saved login token,
  and names generated under the old base URL can't be rated.
- An invalid saved connection, including any saved by 1.0.0-beta.0, is
  discarded and the plugin prompts for a new one.

### Fixed

- A rejected login token is detected from both `401` and `422` responses, so
  missing, expired, or invalid tokens prompt for credentials again instead of
  failing.

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
