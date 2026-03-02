# BR CTF/APP + IBGE Hybrid Build

## What is already staged
- IBGE coordinate base symlink:
  - `/Users/vanyoivanov/Projects/wildfire-nowcast/data/authority/industrial/br/raw_ibge_csvs`

## Authoritative CTF/APP sources (public catalog)
- Dataset id: `168289a4-5813-4186-aefd-ef2ff989cc1b`
- Dataset slug: `pessoas-juridicas-inscritas-no-ctf-app1`
- Catalog query endpoint:
  - `https://dados.gov.br/api/publico/conjuntos-dados/buscar?nome=ctf`
- Example state CSV resource (AC):
  - `https://dadosabertos.ibama.gov.br/dados/CTF/APP/AC/pessoasJuridicas.csv`

## Build command

```bash
cd /Users/vanyoivanov/Projects/wildfire-nowcast
make br-build-hybrid-curated ARGS="\
  --ctf-input /ABSOLUTE/PATH/TO/CTF_APP_CSV_DIR \
  --ibge-dir /Users/vanyoivanov/Projects/wildfire-nowcast/data/authority/industrial/br/raw_ibge_csvs \
  --out /Users/vanyoivanov/Projects/wildfire-nowcast/data/authority/industrial/br/br_ibama_sigel_hybrid_candidate_YYYYMMDD.csv \
  --manifest /Users/vanyoivanov/Projects/wildfire-nowcast/data/authority/industrial/br/br_ibama_sigel_hybrid_candidate_YYYYMMDD_manifest.json \
  --categories 1,2,4,5 \
  --species-codes 6 \
  --allow-municipality-fallback \
  --extracted-at 2026-03-01"
```

## Validate and ingest

```bash
uv run --project ingest python /Users/vanyoivanov/Projects/wildfire-nowcast/scripts/validate_industrial_curated.py \
  --profile br_ibama_sigel_hybrid \
  --input /Users/vanyoivanov/Projects/wildfire-nowcast/data/authority/industrial/br/br_ibama_sigel_hybrid_candidate_YYYYMMDD.csv

make ingest-industrial-authoritative ARGS="--source-profile br_ibama_sigel_hybrid --curated-file /Users/vanyoivanov/Projects/wildfire-nowcast/data/authority/industrial/br/br_ibama_sigel_hybrid_candidate_YYYYMMDD.csv --dry-run"
make ingest-industrial-authoritative ARGS="--source-profile br_ibama_sigel_hybrid --curated-file /Users/vanyoivanov/Projects/wildfire-nowcast/data/authority/industrial/br/br_ibama_sigel_hybrid_candidate_YYYYMMDD.csv"
```
