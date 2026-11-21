# Contributing to AnchorDoc

We welcome pull requests, benchmark additions, and domain adapters!

## Architecture Rules

1. **Zero Domain Hardcoding in `anchordoc/`**: Keep the core engine (`docbase/`, `index/`, `retrieve/`, `extract/`, `adjudicate/`, `report/`, `runtime/`) 100% domain-agnostic. Put domain rules, schemas, and prompts in `adapters/`.
2. **Permissive Core Dependencies**: Core dependencies must remain Apache-2.0 / MIT / BSD compatible. Copyleft accelerators (e.g., PyMuPDF / MinerU) belong strictly in optional extras.
3. **Zero-Leak Hygiene**: Never commit API keys, private IPs, or confidential documents. Always run:
   ```bash
   pytest -q
   python3 tools/scan_sensitive.py
   ```
4. **Reproducible Benchmarks**: Any retrieval or adjudication improvement must be verified on cached public benchmarks (`benchmarks/`) with exact model name and `temperature=0`.
