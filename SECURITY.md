# Security Policy

## Project Status

This repository is a research prototype for experimentation and learning.
It is not production-ready and must not be deployed to public or sensitive environments.

## Supported Versions

No versions are currently supported for production use.

## Reporting a Vulnerability

If you discover a security issue, open a private security advisory in GitHub for this repository.
Do not disclose exploit details in public issues.

## Security Guidance

- Bind services to localhost during development.
- Do not expose services directly to the public internet.
- Do not store secrets, tokens, or credentials in this repository.
- Keep dependencies up to date and run security scans before sharing builds.
- Treat all external input as untrusted and validate/parses messages defensively.
- Use this code only in isolated test environments.
