# EcoTexture AI: Research Narrative and Ablation Plan

This document outlines the scientific framing, architectural advantages, and experimental strategy for the EcoTexture AI hybrid model. It serves as a blueprint for structuring the graduation thesis or research paper.

---

## 1. The Core Research Angle

### Avoid Weak Arguments
Do **not** try to frame the contribution as:
> *"Our model beats state-of-the-art vision transformers."*

Reviewers will immediately counter with larger, pre-trained modern architectures. Instead, frame the work under a realistic and scientifically defensible angle:

> **"Hybrid handcrafted + learned texture fusion improves lightweight waste classification under limited data."**

### Why SIFT + Deep Features Make Sense
TrashNet and similar waste datasets are:
* **Texture-Heavy:** Plastic bottles, crumpled paper, cardboard, and crushed aluminum cans are defined largely by surface texture and deformation patterns.
* **Low Inter-Class Color Separability:** A plastic bottle and a glass jar can look identical in color, but differ in highlights, edges, and surface gradients.
* **Data-Constrained:** Small datasets make it difficult for pure models (especially transformers) to learn invariance from scratch.

By injecting **SIFT (Scale-Invariant Feature Transform)** descriptors directly, we supply a powerful texture prior. The deep CNN backbone focuses on global semantics, while the cross-modal attention block learns how to dynamically fuse the two.

---

## 2. Analysis of the SOTA Model Performance

The latest training run yielded highly promising metrics:

| Metric | Value |
| :--- | :--- |
| **Train Accuracy** | ~96% |
| **Validation Accuracy** | ~90.3% |
| **Total Parameters** | 4.88 M |
| **Trainable Parameters** | 830 K |
| **Frozen Parameters** | 4.05 M |

### Key Takeaways
1. **Not Catastrophically Overfitting:** A gap of ~6% between training and validation accuracy is entirely normal and healthy for a small dataset like TrashNet.
2. **Highly Efficient Attention Block:** The `multi_head_attention_1` layer has only **263,168** parameters. For a fusion mechanism, this is extremely lightweight.
3. **Coherent Bimodal Pipeline:** 
   ```
   EfficientNet ──► Semantic Embedding (1x256) ┐
                                               ├─► Cross-Attention ──► Classifier
   SIFT Hist    ──► Texture Embedding (1x256)  ┘
   ```
   Because the sequence length is only **2** tokens (1 CNN token, 1 SIFT token), this acts as a compact, parameter-efficient relation model rather than standard sequence processing.

---

## 3. The `key_dim` Finding

Comparing two attention setups provides a valuable experimental insight:

* **Version A (`key_dim = 256`):** Overparameterized, higher memory footprint, harder to regularize.
* **Version B (`key_dim = 64`):** Smaller, faster, and generalizes better.

### Reviewer-Friendly Framing
Do not describe this simply as *"reducing parameters."* Frame it as:
> *"We observed that reducing the attention projection dimensionality from 256 to 64 improved generalization while reducing computational complexity, demonstrating that compact cross-modal attention is sufficient for bimodal texture-semantic fusion."*

---

## 4. Model Compression Recommendations (Going Lite)

To strengthen the **"Edge-Deployable / Lightweight AI"** narrative, the architecture can be further compressed into a **Lite** version to run on smart bins or low-power embedded devices.

### Suggested Optimization Path
* **Reduce SIFT Vocabulary:** Shrink $K$ from 300 to 128 (reduces K-Means time and projection layer size).
* **Shrink Dense Layers:** Drop projection layers from 256 to 128.
* **Global Average Pooling 1D:** Replace `Flatten()` after attention with `GlobalAveragePooling1D()` to reduce head parameter count.
* **Replace MHA:** For ultra-lightweight setups, replace Multi-Head Attention with a simple Gated Fusion or Squeeze-and-Excitation block.

| Model Variant | Total Params | Trainable Params | Est. Test Acc |
| :--- | :--- | :--- | :--- |
| **EcoTexture-SOTA** | 4.88 M | 830 K | ~89.4% |
| **EcoTexture-Lite** | ~2.50 M | ~350 K | ~88.8% |

---

## 5. Experimental Plan: What the Paper Needs

To satisfy reviewers, the empirical comparison must isolate the contribution of **your architecture** under controlled conditions.

### A. The Baseline Ablation Table
Answer the question: *"Does the hybrid attention block actually add value beyond standard models?"*

| Model Setup | Backbone | Handcrafted Prior | Fusion Method | Val Acc | Test Acc | Params | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure CNN** | EfficientNetB0 | None | None | - | - | 4.0 M | - |
| **SIFT Only** | None | SIFT (K=300) | None | - | - | - | - |
| **Simple Hybrid** | EfficientNetB0 | SIFT (K=300) | Concatenation | - | - | - | - |
| **EcoTexture SOTA** | EfficientNetB0 | SIFT (K=300) | Cross-Attention (64) | **90.3%**| **89.4%**| **4.8M**| **-** |

### B. Confusion Matrix Analysis
Show exactly where the model struggles (e.g., confusing Glass with Plastic, or Cardboard with Paper). This proves the model is evaluated on practical real-world edge cases.

### C. Efficiency Metrics
Publish both:
* **Trainable vs. Frozen parameters**
* **Inference Latency (FPS)** on CPU vs. GPU, illustrating deployment feasibility on edge devices.
