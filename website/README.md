# Paper project page

English project page for *Flow Matching Reinforcement Learning via SDE Inference*, designed for GitHub Pages under the repository's project path.

## Preview locally

From the repository root:

```bash
python scripts/build_website.py
python scripts/check_website.py
python -m http.server 8765 --bind 127.0.0.1 --directory .site
```

Open `http://127.0.0.1:8765/`. The page is buildless HTML, CSS, and JavaScript; no Node packages, trackers, external fonts, or runtime services are required. The small Python build uses only the standard library. The page self-hosts Manrope, with its SIL Open Font License included.

## Content and publishing

- `site.json` defines the canonical site and repository URLs.
- The build copies the canonical `docs/paper.pdf` and `docs/assets/framework.png` into the Pages artifact, without keeping a second tracked manuscript.
- The experiment images are rendered from the current manuscript. `assets/provenance.json` records its hash, PDF page numbers, and crop bounds. The build rejects a paper/figure mismatch.
- `.github/workflows/pages.yml` builds, checks, and publishes `.site/` when the source changes on `main` or the workflow is triggered manually. The repository's Pages source must be set to GitHub Actions.
- All asset paths are relative so the site works under a project URL such as `/flow_sde/`.

This continues the existing Nerfies-inspired academic project page. Template lineage and the CC BY-SA 4.0 notice remain in the footer; the paper and code retain their respective rights and licenses.

## Interactive recorded-results viewer

The indigo, Manrope typography and rounded experiment panels follow the requested research-lab design. The CIFAR viewer is generated from the nine distributed histories at build time: seed means and population standard deviations are computed for all 80 updates. Controls select Weak/Medium/Strong and the inspected update; they do not simulate new runs. Conditional violations are counted after seed averaging, matching Section 5.4. `?model=weak&update=40` shares a selected view.
