# EcoTexture AI
## Intelligent Hybrid AI for Smart Waste Sorting & Material Recognition
### Supporting Recycling Education and Circular Economy in Egypt

---

**Competition:** Nile University Undergraduate Research Forum (UGRF) 2026  
**Category:** Artificial Intelligence for Sustainability  
**Institution:** Nile University, Egypt  
**Author:** Malak Alaa Mohamed  
**Deadline:** June 7, 2026

---

## Project Title
**EcoTexture AI: Explainable Hybrid Visual Intelligence for Waste Material Recognition and Circular Economy Education in Egypt**

## Tagline
*"See what the AI sees — and learn why it matters."*

---

## One-Paragraph Abstract

EcoTexture AI is an explainable hybrid AI system that classifies 12 categories of waste material (plastic types, metal, glass, paper, cardboard, organic, textile, e-waste, and hazardous) from a single smartphone image. Unlike conventional black-box classifiers, it fuses a CNN (EfficientNetB0) with SIFT texture features through a cross-attention mechanism — the same architecture that achieved 98% audit accuracy in the TimeLens Grand Egyptian Museum recognition system — and visualises its reasoning through GradCAM heatmaps and SIFT keypoint overlays. This transparency is its central educational feature: students do not just receive an answer, they see the texture signatures and spatial attention patterns that led to it. Deployed as a bilingual (Arabic/English) web and mobile application, EcoTexture AI targets Egypt's 21 million tonne/year waste crisis by simultaneously advancing SDG 4 (Quality Education) and SDG 12 (Responsible Consumption and Production), with a five-year scalability roadmap reaching 125,000 Egyptian school students.

---

## Problem Statement

Egypt generates approximately 21 million tonnes of municipal solid waste annually, yet formally recycles only 4–5% — far below the global average of 13%. Organic waste constitutes 55–60% of the total stream, and an estimated one million tonnes of plastic waste enters the Nile each year. This is not primarily a technology gap; it is an education gap. Egypt's 22 million school-age students receive little to no structured, practical waste-sorting education. Contamination rates in existing recycling programmes remain above 30% because citizens — including students — cannot reliably identify materials.

---

## Technical Architecture

```
Input Image
    │
    ├─► EfficientNetB0 (ImageNet → fine-tuned on waste)
    │       └─► Dense(256) projection + LayerNorm
    │
    └─► SIFT BoVW (vocab=300, λ=50, CLAHE-enhanced)
            └─► Dense(256) × 2 + LayerNorm
                        │
            Cross-Attention Fusion (4 heads, d=256)
                        │
            Dense(256) + BatchNorm + Dropout(0.35)
                        │
            Dense(12, softmax)
                        │
            GradCAM + SIFT Heatmaps ← Explainability
```

**Fine-tuning strategy:**
- Stage 1 (15 epochs, lr=1e-3): Head-only, frozen backbone
- Stage 2 (10 epochs, lr=1e-5): Gradual unfreeze of last 30 layers
- Transfer: Contrastive weights from TimeLens (skip_mismatch=True)

---

## SDG Alignment

| SDG | Mechanism |
|-----|-----------|
| SDG 4 — Quality Education | Explainable AI turns every scan into a lesson |
| SDG 9 — Industry & Innovation | Open-source hybrid architecture |
| SDG 11 — Sustainable Cities | Municipal API integration |
| SDG 12 — Responsible Consumption | Recycling guidance with Egyptian cultural context |
| SDG 13 — Climate Action | Quantified CO₂ savings per scan |
| SDG 14 — Life Below Water | Prevents Nile plastic leakage |

---

## WISE Award Criteria

### Innovation
Cross-attention fusion of CNN (global) with SIFT (texture) is a novel architectural contribution. No existing waste AI provides material-specific texture reasoning with educational explainability built in.

### Positive Contribution
- **Direct:** 12-class waste classification with Egyptian-context recycling guidance
- **Educational:** GradCAM + SIFT visualisations as classroom artifacts
- **Societal:** Supports the Zabbaleen informal recycling integration
- **Environmental:** Quantified CO₂ impact per scan, cumulative school dashboard

### Adaptability and Scale
- CPU-only inference — runs on any school tablet
- Fully bilingual Arabic/English
- REST API deployable into Ministry of Education systems
- Re-trainable on Egypt-specific images in <2 hours

---

## Scalability Roadmap

| Milestone | Timeline | Reach |
|-----------|----------|-------|
| Demo + pilot (3 Cairo schools) | Month 1–3 | 750 students |
| Ministry of Education pilot | Month 4–6 | 2,500 students |
| 100-school deployment | Year 2 | 25,000 students |
| MENA regional adaptation | Year 3–4 | 100,000 students |
| 500-school national programme | Year 5 | 125,000 students |

---

*EcoTexture AI: the same hybrid AI that helps visitors of the Grand Egyptian Museum understand 5,000-year-old artifacts now helps students understand the materials in their lunch box.*
