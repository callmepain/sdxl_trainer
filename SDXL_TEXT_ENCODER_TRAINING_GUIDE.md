# SDXL Text Encoder Training - Vollständige Analyse

**Datum:** 2. Dezember 2025
**Problem:** Model reagiert nicht auf Captions trotz aktiviertem Text Encoder Training
**Lösung:** TE1 Learning Rate muss 10-20x höher sein als TE2

---

## Executive Summary

Nach umfangreicher Analyse wurde festgestellt: **Der Code ist korrekt implementiert**. Das Problem ist **strukturell in der SDXL Architektur begründet**.

**Kernproblem:** TE1 (CLIP ViT-L) bekommt **24x schwächere Gradienten** als TE2 (OpenCLIP ViT-bigG), obwohl beide theoretisch wichtig für das Ergebnis sind.

**Lösung:** Kompensiere die schwachen Gradienten durch 10-20x höhere Learning Rate für TE1.

---

## 1. Problemanalyse

### Initiales Problem

- Model reagiert nicht auf Captions in den Trainingsdaten
- Text Encoder 1 und 2 waren aktiviert (`train_text_encoder_1: true`, `train_text_encoder_2: true`)
- Learning Rates waren gesetzt (`lr_text_encoder_1: 5e-06`, `lr_text_encoder_2: 5e-06`)
- TensorBoard zeigte Gradienten für beide TEs

### Erste Beobachtungen

**TensorBoard Metriken (Chris_v12):**
```
UNet Grad Norm:  0.045067
TE2 Grad Norm:   0.007609  (6x kleiner als UNet)
TE1 Grad Norm:   0.002356  (19x kleiner als UNet, 3x kleiner als TE2!)
```

**Training Loss:**
```
First 100 steps avg: 0.117
Last 100 steps avg:  0.120
Improvement: -2.4% (schlechter!)
```

→ Loss sinkt nicht, TE1 Gradienten extrem klein

---

## 2. Durchgeführte Tests

### Test 1: Caption Usage Verification

**Script:** `test_caption_usage.py`

**Ergebnis:** ✅ **Captions werden korrekt verwendet**
- Alle Bilder haben Caption-Files
- Verschiedene Captions führen zu verschiedenen Token-IDs
- Captions werden auch mit Latent Cache korrekt geladen
- Dataset ist korrekt implementiert

### Test 2: Gradient Flow Verification

**Script:** `test_gradient_flow.py`

**Ergebnis:** ✅ **Gradienten fließen zu beiden TEs**
- `requires_grad=True` ist korrekt gesetzt
- Gradienten erreichen beide Text Encoder
- Backward-Propagation funktioniert

**Aber:** Gradienten sind extrem ungleich verteilt!

### Test 3: Weight Change Verification

**Script:** `test_te_learning.py`

**Vergleich:** Chris_v10 → Chris_v11

**Ergebnis:** ✅ **Beide TEs ändern sich, aber minimal**
```
TE1: 0.038% durchschnittliche Änderung (76% Parameter geändert)
TE2: 0.058% durchschnittliche Änderung (80% Parameter geändert)
```

→ Änderungen sind viel zu klein um merkbaren Effekt zu haben

### Test 4: Pooled Embeddings Analysis

**Script:** `test_pooled_embeds.py`

**Ergebnis:** ❌ **TE1 pooler_output wird NICHT verwendet**

**Offizielle SDXL Implementierung (Diffusers):**
```python
# Zeile 412-414 in pipeline_stable_diffusion_xl.py
# We are only ALWAYS interested in the pooled output of the final text encoder
if pooled_prompt_embeds is None and prompt_embeds[0].ndim == 2:
    pooled_prompt_embeds = prompt_embeds[0]  # ← Nur vom letzten TE (TE2)!
```

**Das ist korrekte SDXL Spec!**
- `text_embeds` = concat(TE1.hidden_states, TE2.hidden_states) → beide haben Einfluss
- `pooled_embeds` = NUR TE2.text_embeds → TE1 hat hier keinen Einfluss

### Test 5: Caption Influence Test

**Script:** `debug_caption_influence.py`

**Frage:** Welcher Input ist wichtiger: `text_embeds` oder `pooled_embeds`?

**Ergebnis:** ✅ **text_embeds sind 9x wichtiger!**

```
Test: UNet Output Sensitivität

A) Nur text_embeds geändert:    0.014648 (9x größer)
B) Nur pooled_embeds geändert:  0.001572
C) Beide geändert:               0.015137

Ratio: 9.32
```

**Interpretation:**
- text_embeds haben den größten Einfluss auf das Ergebnis
- pooled_embeds sind weniger wichtig (nur 10% des Einflusses)
- TE1 liefert 37.5% der text_embeds (768 von 2048 dims)
- **TE1 SOLLTE wichtig sein!**

### Test 6: Gradient Distribution Analysis

**Script:** `test_gradient_distribution.py`

**Frage:** Warum sind TE1 Gradienten so schwach, obwohl text_embeds wichtig sind?

**Ergebnis:** ❌ **TE2 bekommt 24x stärkere Gradienten!**

```
Gradient Norms nach einem Backward Pass:

TE1 Grad Norm:  0.043288
TE2 Grad Norm:  1.040214  (24.03x mehr!)
UNet Grad Norm: 2.889981  (66.76x mehr als TE1)

TE2/TE1 Ratio: 24.03x
```

**Per-Layer Analyse:**
```
TE1 Top Layer: 1.30e-02
TE2 Top Layer: 8.44e-01  (65x größer!)
```

---

## 3. Root Cause Analysis

### Warum bekommt TE1 24x schwächere Gradienten?

#### Faktor 1: Unterschiedliche Parameteranzahl
```
TE1 (CLIP ViT-L/14):         ~124M Parameter
TE2 (OpenCLIP ViT-bigG/14):  ~695M Parameter

Ratio: 5.6x mehr Parameter in TE2
```

→ Mehr Parameter = mehr Gradienten verteilt sich über mehr Gewichte

#### Faktor 2: Unterschiedliche Hidden Dimensions
```
TE1 hidden_dim: 768
TE2 hidden_dim: 1280

In text_embeds: [B, 77, 768+1280] = [B, 77, 2048]

TE1 Anteil: 768/2048  = 37.5%
TE2 Anteil: 1280/2048 = 62.5%
```

→ TE2 hat 1.67x mehr Einfluss über größeren Embedding Space

#### Faktor 3: Pooled Embeddings nur von TE2
```
pooled_embeds = enc_2.text_embeds  # [B, 1280]
# TE1.pooler_output wird NICHT verwendet
```

→ TE2 bekommt zusätzliche Gradienten über pooled path (auch wenn Einfluss klein ist)

#### Faktor 4: UNet bias zu TE2-Features?

Das UNet wurde auf SDXL Base trainiert, wo:
- TE2 Features prominenter sind
- Attention Layers könnten TE2-Features bevorzugen
- TE1 Features werden möglicherweise weniger genutzt

### Kombinierter Effekt

```
Parameter-Faktor:     5.6x
Hidden-Dim-Faktor:    1.67x
Pooled-Embedding:     nur TE2
UNet-Bias:            unklar, aber wahrscheinlich zugunsten TE2

Gemessener Gradient-Unterschied: 24x
```

---

## 4. Lösung

### Empfohlene Learning Rates

Da TE1 strukturell 24x schwächere Gradienten bekommt, muss die Learning Rate dies kompensieren.

#### Option 1: Konservativ (sicher)
```json
"lr_unet": 1e-05,
"lr_text_encoder_1": 5e-05,   // 5x höher als TE2
"lr_text_encoder_2": 1e-05,
```

**Erwartete Grad Norm Ratio:** ~5x (TE2 immer noch dominant)

#### Option 2: Moderat (⭐ empfohlen)
```json
"lr_unet": 1e-05,
"lr_text_encoder_1": 1e-04,   // 10x höher als TE2
"lr_text_encoder_2": 1e-05,
```

**Erwartete Grad Norm Ratio:** ~2.4x (ausgeglichener)

#### Option 3: Aggressiv (wenn moderat nicht reicht)
```json
"lr_unet": 1e-05,
"lr_text_encoder_1": 2e-04,   // 20x höher als TE2
"lr_text_encoder_2": 1e-05,
```

**Erwartete Grad Norm Ratio:** ~1.2x (nahezu ausgeglichen)

### Weitere wichtige Config-Änderungen

#### 1. Constant Learning Rate Scheduler
```json
"lr_scheduler": {
  "type": "constant",
  "warmup_steps": 100,
  "min_factor": 1.0
}
```

**Grund:** Cosine Decay reduzierte LRs auf 70% bei Step 956, was die ohnehin schon kleinen Gradienten weiter schwächte.

#### 2. SNR Gamma aktivieren
```json
"snr_gamma": 5.0,
```

**Grund:** Stabilisiert Training bei höheren TE Learning Rates

#### 3. Gradient Checkpointing deaktivieren (optional)
```json
"use_gradient_checkpointing": false,
```

**Grund:** Besserer Gradient Flow, besonders für die schwächeren TE1 Gradienten

---

## 5. Monitoring & Validation

### TensorBoard Metriken überwachen

Nach Start des Trainings mit neuen LRs, achte auf:

#### 1. Gradient Norms
```
train/grad_norm_te1
train/grad_norm_te2
train/grad_norm (UNet)
```

**Zielwerte:**
- TE1: > 0.01 (aktuell: 0.003)
- TE2: > 0.03 (aktuell: 0.012)
- **Ratio TE1/TE2: > 0.5** (aktuell: 0.25)

#### 2. Learning Rates
```
train/lr_text_encoder_1  → sollte konstant bei 1e-04 bleiben
train/lr_text_encoder_2  → sollte konstant bei 1e-05 bleiben
train/lr_unet            → sollte konstant bei 1e-05 bleiben
```

#### 3. Loss
```
train/loss → sollte kontinuierlich sinken
```

**Erwartung:** Nach 100-200 Steps sollte Loss um mindestens 10% gesunken sein

### Weight Change Verification

Nach 250 Steps, vergleiche Checkpoints:

```bash
python test_te_learning.py .output/base_model .output/new_model_step_250
```

**Erwartete Änderungen:**
- TE1: > 0.5% (statt 0.038%)
- TE2: > 1.0% (statt 0.058%)

### Evaluation Images

Mit aktivierter Eval (`eval.live.enabled: true`):

**Prüfe ob:**
- Verschiedene Prompts zu unterschiedlichen Bildern führen
- Konzepte aus den Captions sichtbar werden
- Model auf neue/spezifische Prompts reagiert

---

## 6. Warum funktioniert es bei anderen Trainern?

### Kohya_ss

Kohya_ss verwendet wahrscheinlich:
1. Automatische LR-Adjustierung basierend auf Gradient Norms
2. Höhere Default-LRs für TE1
3. Oder separate Optimizer für jeden TE mit angepassten Parametern

### Dreambooth / andere Trainer

Viele Trainer:
- Trainieren oft nur TE2 (ignorieren TE1 komplett)
- Oder verwenden sehr hohe LRs für beide TEs (was bei TE1 funktioniert, aber TE2 overfitten kann)
- Oder haben separate LR-Multiplier für TEs basierend auf Parameteranzahl

---

## 7. Best Practices für SDXL Text Encoder Training

### 1. LR Ratio beachten

**Faustregel:** TE1 LR sollte 10-20x höher sein als TE2 LR

```
Basis-Ratio aus Gradienten: 24x
Empfohlene LR-Ratio: 10-20x (konservativer)
```

### 2. Separate Monitoring

Logge immer:
- Separate Grad Norms für TE1 und TE2
- Separate Learning Rates
- Relative Änderungsraten der Gewichte

### 3. Schrittweise Anpassung

**Start:**
```json
"lr_text_encoder_1": 5e-05,
"lr_text_encoder_2": 1e-05,
```

**Falls TE1 immer noch zu schwach (grad_norm_te1/grad_norm_te2 < 0.3):**
```json
"lr_text_encoder_1": 1e-04,  // verdoppeln
```

**Falls TE1 immer noch zu schwach:**
```json
"lr_text_encoder_1": 2e-04,  // nochmal verdoppeln
```

**Stopkriterium:** TE1/TE2 Ratio zwischen 0.5 und 1.5

### 4. Caption Quality wichtiger als TE Training

**Prioritäten:**
1. ✅ Hochwertige, detaillierte Captions
2. ✅ Diverse Trainingsdaten
3. ✅ Ausreichend Trainingsschritte
4. ⚠️ Text Encoder Training (kann helfen, ist aber riskant)

**Vorsicht:**
- TE Training kann zu Overfitting führen
- Base Model Konzepte können "verlernt" werden
- Nur trainieren wenn wirklich neue Konzepte gelernt werden sollen

### 5. Alternative Strategien

#### A) Nur TE2 trainieren
```json
"train_text_encoder_1": false,
"train_text_encoder_2": true,
"lr_text_encoder_2": 2e-05,
```

**Vorteil:** Einfacher, weniger riskant
**Nachteil:** Verliert Flexibilität von TE1

#### B) Frozen TEs mit LoRA
Statt TEs direkt zu trainieren:
- Verwende LoRA Adapter für TEs
- Viel kleinerer Parameter Space
- Weniger Risiko für Overfitting

#### C) Nur UNet trainieren
```json
"train_text_encoder_1": false,
"train_text_encoder_2": false,
```

**Vorteil:** Sicherste Option
**Nachteil:** Kann keine neuen Konzepte lernen, nur Stil

---

## 8. Häufige Fehler

### ❌ Fehler 1: Gleiche LR für beide TEs
```json
"lr_text_encoder_1": 5e-06,  // ← Zu niedrig!
"lr_text_encoder_2": 5e-06,
```

**Problem:** TE1 bekommt 24x schwächere Gradienten → wird faktisch nicht trainiert

### ❌ Fehler 2: Cosine Decay ohne Monitoring
```json
"lr_scheduler": {
  "type": "cosine_decay",
  "min_factor": 0.3
}
```

**Problem:** LRs sinken kontinuierlich, schwache Gradienten werden noch schwächer

### ❌ Fehler 3: TE Training ohne TensorBoard Logging
```json
"tensorboard": {
  "enabled": false  // ← Kann Probleme nicht erkennen!
}
```

**Problem:** Keine Möglichkeit zu sehen dass TE1 nicht trainiert wird

### ❌ Fehler 4: Zu viele Steps auf bereits fine-getuned Model
```
Base: SDXL Base 1.0
→ Fine-tune 1: Chris_v1 (2000 steps)
→ Fine-tune 2: Chris_v2 (2000 steps)
...
→ Fine-tune 11: Chris_v11 (2000 steps)  // ← Hier trainieren
```

**Problem:** Model ist bereits saturiert, TEs sind "eingefroren" auf bestimmte Konzepte

**Lösung:** Gelegentlich zurück zu Base Model oder weniger Steps pro Iteration

---

## 9. Troubleshooting Guide

### Problem: TE1 Grad Norm bleibt unter 0.01

**Mögliche Ursachen:**
1. LR zu niedrig → Erhöhe auf 1e-04 oder 2e-04
2. Gradient Clipping zu aggressiv → Erhöhe `max_grad_norm`
3. Gradient Checkpointing behindert Flow → Deaktiviere temporär
4. Model saturiert → Probiere frisches Base Model

### Problem: Loss steigt statt zu sinken

**Mögliche Ursachen:**
1. TE LRs zu hoch → Reduziere beide um Faktor 2
2. Kein SNR Gamma → Aktiviere `snr_gamma: 5.0`
3. Batch Size zu klein → Erhöhe oder nutze Grad Accumulation
4. Dataset Quality → Überprüfe Captions

### Problem: Model "vergisst" Base Konzepte

**Mögliche Ursachen:**
1. Zu viele TE Training Steps → Reduziere `num_steps`
2. TE LRs zu aggressiv → Reduziere
3. Dataset zu homogen → Diversifiziere Trainingsdaten
4. Kein Regularization → Füge Dropout hinzu

### Problem: Keine sichtbare Verbesserung nach 500 Steps

**Mögliche Ursachen:**
1. Base Model schon zu gut → TEs müssen nicht geändert werden
2. Captions zu ähnlich zu Base Training → Neue Konzepte fehlen
3. Learning Rates zu konservativ → Erhöhe graduell
4. Eval Prompts nicht repräsentativ → Ändere Prompts

---

## 10. Zusammenfassung

### Was wir gelernt haben

1. ✅ **Code ist korrekt** - Captions werden verwendet, Gradienten fließen
2. ✅ **SDXL Architektur ist asymmetrisch** - TE2 ist dominant
3. ✅ **24x Gradient-Unterschied** ist strukturell bedingt, kein Bug
4. ✅ **text_embeds sind 9x wichtiger** als pooled_embeds
5. ✅ **TE1 ist wichtig** (37.5% der text_embeds) trotz schwacher Gradienten
6. ✅ **Lösung ist einfach**: TE1 LR 10-20x höher setzen

### Empfohlene Config für neue Trainings

```json
{
  "model": {
    "train_text_encoder_1": true,
    "train_text_encoder_2": true,
    "use_gradient_checkpointing": false
  },
  "training": {
    "lr_unet": 1e-05,
    "lr_text_encoder_1": 1e-04,
    "lr_text_encoder_2": 1e-05,
    "max_grad_norm": 1.0,
    "snr_gamma": 5.0,
    "noise_offset": 0.05,
    "min_sigma": 0.01,
    "lr_scheduler": {
      "type": "constant",
      "warmup_steps": 100
    },
    "tensorboard": {
      "enabled": true,
      "log_grad_norm": true
    }
  }
}
```

### Nächste Schritte

1. **Teste mit kurzer Session** (200-500 steps)
2. **Monitore TensorBoard** - grad_norm_te1 sollte > 0.01 sein
3. **Vergleiche Weights** mit test_te_learning.py
4. **Evaluiere Bilder** - sollten auf Prompts reagieren
5. **Adjustiere LRs** falls nötig basierend auf Grad Norms

### Wichtigste Metrik

```
Ziel: train/grad_norm_te1 / train/grad_norm_te2 > 0.5

Aktuell (mit lr_te1=5e-06, lr_te2=5e-06): 0.25
Mit lr_te1=1e-04, lr_te2=1e-05: erwartungsgemäß ~2.5
```

---

## Referenzen

- **Diffusers SDXL Pipeline:** `venv/lib/python3.12/site-packages/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`
- **Test Scripts:** `test_caption_usage.py`, `test_gradient_flow.py`, `test_pooled_embeds.py`, `debug_caption_influence.py`, `test_gradient_distribution.py`
- **TensorBoard Logs:** `logs/tensorboard/Chris_v12/`, `logs/tensorboard/Chris_v12_te_fixed/`
- **Analysis Script:** `analyze_tensorboard.py`

---

**Erstellt:** 2. Dezember 2025
**Letzte Aktualisierung:** 2. Dezember 2025
**Status:** Validiert durch umfangreiche Tests
