# Security Policy

## Supported versions

Only the latest version of `removed_bg.py` receives security patches. Older commits are not back-ported.

| Version | Supported |
|---------|-----------|
| Latest (`main`) | ✅ |
| Older commits | ❌ |

---

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

If you discover a security issue, report it privately via one of the following channels:

- **GitHub Private Vulnerability Reporting** — use the [Security tab → Report a vulnerability](https://github.com/Biraj2004/removed_bg/security/advisories/new) button on the repository page.
- **Email** — if GitHub's private reporting is unavailable, contact the maintainer directly through the email address listed on their [GitHub profile](https://github.com/Biraj2004).

Please include as much detail as possible:

- A clear description of the vulnerability and its potential impact.
- Steps to reproduce, including any sample inputs or commands.
- Python version, OS, and dependency versions (`pip freeze` output).
- Any suggested fix or mitigation, if you have one.

You can expect an acknowledgement within **72 hours** and a status update (fix ETA or decision to decline) within **7 days**.

---

## Security considerations

### Local execution model

`removed_bg.py` is a fully **local, offline** tool. It does not:

- transmit images, metadata, or any user data to a remote server.
- make network requests during normal operation (beyond the one-time model download described below).
- store any credentials or API keys.

### One-time model download (AI method)

On first use of `--method ai`, the U2-Net ONNX model (~170 MB) is downloaded from `https://github.com/danielgatis/rembg` and cached at `~/.u2net/u2net.onnx`.

- The download is handled entirely by the `rembg` library.
- To verify the downloaded model manually, compare its SHA-256 hash against the value published in the [rembg releases](https://github.com/danielgatis/rembg/releases).
- In air-gapped environments, pre-download the model on a trusted machine and copy it to `~/.u2net/` before running the script.

### Input validation

The script validates input file extensions against a fixed allowlist (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`) and rejects unsupported formats. Output paths are resolved relative to the current working directory; no paths outside the working directory are written to by default.

### Dependency supply-chain

Dependencies are pinned with minimum versions in `requirements.txt`. Keep them up to date to receive upstream security patches:

```bash
pip install --upgrade -r requirements.txt
```

Monitor advisories for the following packages:

| Package | Advisory source |
|---------|----------------|
| `Pillow` | [GitHub Advisories](https://github.com/python-pillow/Pillow/security) |
| `numpy` | [GitHub Advisories](https://github.com/numpy/numpy/security) |
| `rembg` | [GitHub Advisories](https://github.com/danielgatis/rembg/security) |
| `onnxruntime` | [GitHub Advisories](https://github.com/microsoft/onnxruntime/security) |

You can automate this with `pip-audit`:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

### Running in untrusted environments

- Do not run `removed_bg.py` as `root` or with elevated privileges.
- When processing images from untrusted sources, consider sandboxing (e.g., a container or virtual environment) as a precaution against malformed image files that could exploit parser vulnerabilities in Pillow or the ONNX runtime.

---

## Disclosure policy

This project follows **coordinated disclosure**: vulnerabilities are kept private until a patch is available, at which point a GitHub Security Advisory is published describing the issue and the fix.

---

## Acknowledgements

We appreciate responsible security research. Reporters who follow this policy will be credited in the Security Advisory unless they prefer to remain anonymous.
