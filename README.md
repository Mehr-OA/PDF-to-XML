# PDF-to-XML and Plasma Physics Annotation Pipeline

This repository provides a pipeline for:

1. Collecting items from a **RENATE (DSpace)** collection
2. Downloading their PDFs
3. Converting PDFs to **JATS XML** using **GROBID**
4. Uploading the XML back to the item
5. Running **plasma physics NER** on relevant articles
6. Enriching item metadata with extracted entities

The pipeline is designed to be executed by a human operator using a simple shell script.

---

## Overview of the Pipeline

```
RENATE collection
      │
      ▼
Fetch items via API
      │
      ▼
Download PDFs
      │
      ▼
GROBID (PDF → TEI XML)
      │
      ▼
TEI → JATS conversion
      │
      ▼
Upload JATS XML to item
      │
      ▼
Plasma-physics detection
      │
      ▼
NER annotation (BERT-CRF)
      │
      ▼
Add extracted entities to item metadata
```

## Repository Structure

```
├── app.py
├── run_pipeline.sh
├── FileService.py
├── AnnotationService.py
├── AuthService.py
├── config_loader.py
├── trained_models/
│ ├── DiagnosticDevice/
│ ├── PlasmaSource/
│ └── ...
├── tei_to_jats.xslt
├── config.yml
└── README.md
```

---

## Requirements

### System Requirements
- Python **3.9+**
- Linux or macOS (tested with bash shell)

### Python Dependencies
Install using:

bash
pip install -r requirements.txt

requests
lxml
torch
transformers
pyyaml


### GROBID Setup

This pipeline requires a running GROBID server.

#### Option 1 — Docker (recommended)

docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2

GROBID will be available at:

http://localhost:8070

The pipeline expects:

GROBID_API_URL = "http://localhost:8070/api/processFulltextDocument"

```
COLLECTION_ITEMS_ENDPOINT: "https://oa.tib.eu/renate/server/api/discover/search/objects?scope="
UPLOAD_BITSTREAMS_ENDPOINT: "https://oa.tib.eu/renate/server/api/core/bundles/{bundle_uuid}/bitstreams"
UPDATE_ITEMS_METADATA: "https://oa.tib.eu/renate/server/api/core/items/{item_uuid}"
```

Authentication settings are handled in:
```
AuthService.py
```

### Running the Pipeline

#### Step 1 — Make the script executable
chmod +x run_pipeline.sh

#### Step 2 — Run the pipeline
./run_pipeline.sh

###Incremental Processing

The pipeline stores the last run timestamp in:

```
last_run.json
```

On the next execution:

Only newly modified items are processed.

