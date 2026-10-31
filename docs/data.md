# Data

11 public-domain FAA handbooks, 3,831 pages, 708 MB. Files live in `data/pdfs/` (not in git).

**Download:** [Google Drive](https://drive.google.com/drive/folders/1j5tXmtN4bHrlV_OSb0Tz6n32vqZU1Ofo?usp=drive_link) (PDFs + processed pages/figures + chunks). Unzip into `data/`.

| # | File | Book | Pages | Notes |
|---|---|---|---|---|
| 01 | `01_aviation_instructors_handbook.pdf` | Aviation Instructor's Handbook | 228 | |
| 02 | `02_glider_flying_handbook.pdf` | Glider Flying Handbook | 276 | |
| 03 | `03_weight_shift_control_handbook.pdf` | Weight-Shift Control Aircraft Flying Handbook | 280 | |
| 04 | `04_instrument_procedures_handbook.pdf` | Instrument Procedures Handbook | 312 | approach charts |
| 05 | `05_instrument_flying_handbook.pdf` | Instrument Flying Handbook | 371 | |
| 06 | `06_helicopter_flying_handbook.pdf` | Helicopter Flying Handbook | 203 | 176 MB |
| 07 | `07_rotorcraft_flying_handbook.pdf` | Rotorcraft Flying Handbook (2000) | 207 | |
| 08 | `08_pilots_handbook_aeronautical_knowledge.pdf` | Pilot's Handbook of Aeronautical Knowledge | 522 | |
| 09 | `09_aerodynamics_for_naval_aviators_1965.pdf` | Aerodynamics for Naval Aviators (1965) | 434 | **scanned**, bad old OCR layer -> re-OCR |
| 10 | `10_mc4_parachute_technical_manual.pdf` | MC-4 Ram Air Parachute Technical Manual | 484 | one undecodable JPX image (p. 420) |
| 11 | `11_aviation_weather_handbook.pdf` | Aviation Weather Handbook | 514 | |

## Sources

| File | URL |
|---|---|
| 01 | https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/aviation_instructors_handbook/aviation_instructors_handbook.pdf |
| 02 | https://www.faa.gov/sites/faa.gov/files/Glider-Flying-Handbook.pdf |
| 03 | https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/FAA-H-8083-5.pdf |
| 04 | https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/instrument_procedures_handbook/FAA-H-8083-16B.pdf |
| 05 | https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/FAA-H-8083-15B.pdf |
| 06 | https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/helicopter_flying_handbook/helicopter_flying_handbook.pdf |
| 07 | https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/faa-h-8083-21.pdf |
| 08 | https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/faa-h-8083-25c.pdf |
| 09 | https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/00-80T-80.pdf |
| 10 | https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/mc4_technical_manual.pdf |
| 11 | https://www.faa.gov/sites/faa.gov/files/FAA-H-8083-28B.pdf |

Download all (file names must match, ingestion uses the file stem as `pdf_id`):

```bash
mkdir -p data/pdfs && cd data/pdfs
curl -fL -A "Mozilla/5.0" -o 01_aviation_instructors_handbook.pdf "<URL 01>"
# ... one line per row above
```

## Generated data (all under `data/`, git-ignored)

| Path | Produced by | Content |
|---|---|---|
| `processed/<book>/pages.jsonl` | ingestion | one line per page: markdown text + metadata + figures |
| `processed/<book>/figures/*.png` | ingestion | figure crops (110 dpi) |
| `chunks/<book>.jsonl` | chunking | one line per chunk: text + metadata |

Totals: 3,831 pages, 4,106 chunks, 2,761 figures.
