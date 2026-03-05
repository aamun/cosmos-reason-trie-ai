# Cosmos Reason 2: Traffic Incident Evidence Agent (TRiE-AI)

A reproducible Physical AI project for **traffic incident understanding** and **evidence-style reporting** using **NVIDIA Cosmos Reason 2**.

**What it does**

* Takes a traffic video clip (dashcam/CCTV)
* Produces a **structured evidence report** (JSON) with:
  * actors, event timeline, causal chain, and risk assessment

## Cosmos Reason 2 

NVIDIA Cosmos Reason 2 is purpose-built for Physical AI reasoning model.

## Architecture

```mermaid
flowchart TD

A[Traffic Video Input\nMP4] --> B[Frame Sampling\nUniform Sampling\nN frames]
B --> C[Frame Encoding\nFrame Index\nTimestamp\nVisual Description]
C --> D[Cosmos Reason 2\nVision-Language Reasoning Model]
D --> E[Prompt Template\nTraffic Incident Analyst]
E --> F[Structured JSON Report]

F --> G1[Summary]
F --> G2[VQA Answers\nDay/Night\nWeather\nEgo Involved]
F --> G3[Actors List]
F --> G4[Events Timeline]
F --> G5[Risk Assessment]
F --> G6[Uncertainties]
F --> H[Stored Reports\n1 JSON per Video]
H --> I[Evaluation Pipeline]
I --> J[Crash-1500 Parser\nGround Truth Labels]
H --> K[VQA Extraction\nPredicted Answers]
J --> L[VQA Evaluation]
K --> L
L --> M[Metrics Output]
M --> N[Accuracy\nUnknown Rate\nConsistency Metrics]
```