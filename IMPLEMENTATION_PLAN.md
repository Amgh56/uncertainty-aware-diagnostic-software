# SafeDx — Scalable Multi-Model Platform: Full Implementation Plan

---

## Section 1 — High-Level Product Architecture

### Core Concept

SafeDx evolves from a **single hardcoded model system** into a **scalable, model-driven diagnostic platform** where:

- **Developers/Researchers** upload, calibrate, validate, and optionally release models
- **Clinicians/Doctors** consume validated, released models for patient diagnosis
- **The developer community** can browse and reuse calibrated models shared by others

The platform has three surfaces:

| Surface | Audience | Purpose |
|---------|----------|---------|
| **Developer Workspace** | Researchers, ML engineers | Upload → Calibrate → Validate → Publish models |
| **Model Library** | Developers/Researchers | Browse, inspect, and download community-shared calibrated models |
| **Clinical Console** | Doctors, Clinicians | Select a validated model → Upload patient image → Get uncertainty-aware diagnosis |

### The Bridge: Published Model Package

The central new concept is the **Published Model Package** — a self-contained, versioned unit that bundles everything needed to run inference with full calibration and risk control. This replaces the current loose coupling between a `.pth` file, a `lamhat.json`, and hardcoded disease labels.

A Published Model Package contains:

```
┌─────────────────────────────────────────┐
│         Published Model Package         │
├─────────────────────────────────────────┤
│  Model artifact (.pth / .pt)            │
│  Artifact type (pytorch / torchscript)  │
│  Preprocessing config (JSON)            │
│  Label schema (ordered label names)     │
│  Selected alpha                         │
│  Computed lamhat                        │
│  Validation verdict + metrics           │
│  Metadata:                              │
│    - name, description                  │
│    - modality (e.g. "Chest X-Ray")      │
│    - intended use                       │
│    - version string                     │
│    - owner/creator reference            │
│  Visibility flags                       │
│  Consent record                         │
└─────────────────────────────────────────┘
```

**I agree with your instinct** that this "published model package" abstraction is the correct bridge. The alternative — exposing raw calibration job outputs directly — would leak implementation details to clinicians and make the system fragile. The package abstraction gives you:

1. **Clean separation**: calibration internals vs. consumable product
2. **Versioning**: same model can be re-calibrated and re-published as a new version
3. **Access control**: visibility is a property of the package, not the raw artifacts
4. **Self-containment**: everything needed for inference lives in one logical unit

### System Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          SafeDx Platform                             │
│                                                                      │
│  ┌─────────────────────┐          ┌─────────────────────────────┐   │
│  │  Developer Workspace │          │     Clinical Console        │   │
│  │                      │          │                             │   │
│  │  Upload Model        │          │  Select Published Model ──────► │
│  │       │              │          │       │                     │   │
│  │       ▼              │          │       ▼                     │   │
│  │  Run Calibration     │          │  Upload Patient Image      │   │
│  │       │              │          │       │                     │   │
│  │       ▼              │          │       ▼                     │   │
│  │  Validate Results    │          │  Run Inference              │   │
│  │       │              │          │  (uses package's artifact,  │   │
│  │       ▼              │          │   lamhat, alpha, labels)    │   │
│  │  Publish Model ─────────────►  │       │                     │   │
│  │  (with consent)      │   ┌──►  │       ▼                     │   │
│  │                      │   │     │  Dynamic Results UI         │   │
│  └─────────────────────┘   │     │  (model's own labels)       │   │
│                             │     └─────────────────────────────┘   │
│  ┌─────────────────────┐   │                                        │
│  │  Model Library       │   │                                        │
│  │  (Community Models) ◄────┘                                        │
│  │  Browse / Download   │                                            │
│  └─────────────────────┘                                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Section 2 — Recommended Lifecycle / State Flow

### Model Lifecycle

A model moves through these states:

```
UPLOADED ──► CALIBRATING ──► CALIBRATED ──► VALIDATED ──► PUBLISHED
                │                               │            │
                ▼                               │            ├── PRIVATE
            FAILED                              │            ├── CLINICIAN
                                                │            ├── COMMUNITY
                                                │            └── CLINICIAN_AND_COMMUNITY
                                                │
                                                ▼
                                          (not publishable
                                           if verdict = "unreliable")
```

### Detailed State Descriptions

| State | Meaning | Who sees it |
|-------|---------|-------------|
| **UPLOADED** | Model file received, job queued | Developer only |
| **CALIBRATING** | Background calibration job running | Developer only |
| **FAILED** | Calibration job errored out | Developer only |
| **CALIBRATED** | Calibration complete, lamhat computed, job status = DONE | Developer only |
| **VALIDATED** | Developer has viewed validation results; system has assigned a verdict | Developer only |
| **PUBLISHED** | Developer chose to release; consent recorded; Published Model Package created | Depends on visibility |

### Transition Rules

1. **UPLOADED → CALIBRATING**: Automatic when background task starts
2. **CALIBRATING → CALIBRATED**: Automatic when job completes successfully
3. **CALIBRATING → FAILED**: Automatic on error
4. **CALIBRATED → VALIDATED**: When developer views validation page and validation data is generated (this already happens via the validation endpoint)
5. **VALIDATED → PUBLISHED**: When developer explicitly publishes with metadata + consent. The system should **warn but not block** publishing models with "review" verdict. It should **block** publishing models with "unreliable" verdict (they need re-calibration).

### Key Design Decision: Validation Verdict Gates Publishing

| Verdict | Can publish for clinician use? | Can publish for community use? |
|---------|-------------------------------|-------------------------------|
| **good** | Yes | Yes |
| **review** | Yes (with warning) | Yes (with warning) |
| **unreliable** | No | No |

This is a pragmatic gate for the prototype. In future work, "review" verdicts could require additional expert approval before clinician release.

---

## Section 3 — Recommended Data Model / Database Entities

### Entity Relationship Overview

```
Doctor (existing)
  │
  ├──► Patient (existing)
  │       │
  │       └──► Prediction (existing, modified)
  │               │
  │               └──► published_model_id (NEW FK)
  │
  ├──► CalibrationJob (existing, modified)
  │       │
  │       └──► PublishedModel (NEW)
  │               │
  │               └──► labels_json, artifact_path, alpha, lamhat, visibility, ...
  │
  └──► (developer owns jobs + published models)
       (clinician owns patients + predictions)
```

### Entity 1: Doctor (EXISTING — no changes)

**Purpose**: Represents any authenticated user (clinician or developer).

| Field | Type | Notes |
|-------|------|-------|
| id | Integer PK | Auto-increment |
| email | String | Unique |
| hashed_password | String | bcrypt |
| full_name | String | Display name |
| role | Enum | CLINICIAN / DEVELOPER |
| created_at | DateTime | |

No changes needed. The role field already distinguishes clinicians from developers.

---

### Entity 2: Patient (EXISTING — no changes)

**Purpose**: Medical record linked to a clinician.

| Field | Type | Notes |
|-------|------|-------|
| id | Integer PK | |
| mrn | String | Medical record number |
| first_name | String | |
| last_name | String | |
| doctor_id | FK → Doctor | |
| created_at | DateTime | |

No changes needed.

---

### Entity 3: CalibrationJob (EXISTING — minor additions)

**Purpose**: Tracks a calibration run initiated by a developer.

| Field | Type | Notes |
|-------|------|-------|
| id | UUID PK | Existing |
| developer_id | FK → Doctor | Existing |
| status | Enum | QUEUED/RUNNING/DONE/FAILED (existing) |
| model_filename | String | Existing |
| config_filename | String | Existing (nullable) |
| dataset_filename | String | Existing |
| alpha | Float | Existing |
| result_json | JSON | Existing (lamhat result) |
| error_message | Text | Existing (nullable) |
| created_at | DateTime | Existing |
| completed_at | DateTime | Existing (nullable) |
| **validation_verdict** | **String (nullable)** | **NEW** — "good" / "review" / "unreliable", set when validation runs |
| **is_published** | **Boolean** | **NEW** — default False, set True when a PublishedModel is created from this job |

**Relationships**:
- belongs to Doctor (developer_id)
- has zero or one PublishedModel (one-to-one, optional)

**Rationale for additions**:
- `validation_verdict`: Currently computed on-the-fly but not persisted. Persisting it allows querying "which jobs are publishable?" without re-computing.
- `is_published`: Prevents publishing the same calibration job twice, and allows the UI to show which jobs have already been published.

---

### Entity 4: PublishedModel (NEW — core new entity)

**Purpose**: The self-contained "Published Model Package" — the bridge between developer calibration and clinician inference.

| Field | Type | Notes |
|-------|------|-------|
| id | UUID PK | Unique identifier |
| calibration_job_id | FK → CalibrationJob | Which job produced this (unique, one-to-one) |
| developer_id | FK → Doctor | Owner/creator |
| **Identity** | | |
| name | String(150) | Display name (e.g., "CheXpert Chest X-Ray v2") |
| description | Text | Short description of what the model does |
| version | String(20) | Semantic version (e.g., "1.0.0") |
| **Classification** | | |
| modality | String(100) | e.g., "Chest X-Ray", "Dermatoscopy", "Retinal OCT" |
| intended_use | Text | What the model is designed for |
| **Technical Package** | | |
| artifact_path | String | Path to stored model file on server |
| artifact_type | String(20) | "pytorch" / "torchscript" (extensible) |
| config_json | JSON | Preprocessing config (width, height, pixel_mean, pixel_std, etc.) |
| labels_json | JSON | Ordered array of label names (e.g., `["Cardiomegaly", "Edema", ...]`) |
| num_labels | Integer | Number of labels (denormalized for convenience) |
| **Calibration Outputs** | | |
| alpha | Float | Selected alpha value |
| lamhat | Float | Computed conformal threshold |
| lamhat_result_json | JSON | Full lamhat.json content |
| **Validation Outputs** | | |
| validation_verdict | String | "good" / "review" |
| validation_metrics_json | JSON | FNR, avg_set_size, n_samples, n_positive, etc. |
| **Visibility & Release** | | |
| visibility | Enum | PRIVATE / CLINICIAN / COMMUNITY / CLINICIAN_AND_COMMUNITY |
| is_active | Boolean | Default True. Can be deactivated by developer or admin |
| **Consent** | | |
| consent_given_at | DateTime | When developer agreed to release terms |
| consent_text_hash | String(64) | SHA-256 of the consent text version they agreed to |
| **Timestamps** | | |
| created_at | DateTime | When published |
| updated_at | DateTime | Last modification |

**Relationships**:
- belongs to Doctor (developer_id)
- belongs to CalibrationJob (calibration_job_id, one-to-one)
- has many Predictions (predictions that used this model)

**Key Constraints**:
- `calibration_job_id` is unique (one job → at most one published model)
- `visibility != PRIVATE` requires `consent_given_at IS NOT NULL`
- Only models with `validation_verdict IN ('good', 'review')` can be published

**Indexes**:
- `(visibility, is_active)` — for efficient clinician/community queries
- `developer_id` — for "my published models" queries
- `modality` — for filtering in the model library

---

### Entity 5: Prediction (EXISTING — one addition)

**Purpose**: A single diagnostic prediction for a patient.

| Field | Type | Notes |
|-------|------|-------|
| ... | ... | All existing fields unchanged |
| **published_model_id** | **FK → PublishedModel (nullable)** | **NEW** — which published model was used for this prediction |

**Rationale**:
- Links each prediction to the specific model that generated it
- Nullable for backward compatibility with existing predictions (those used the hardcoded model)
- Enables audit trail: "which model produced this diagnosis?"
- Enables per-model performance tracking in future

---

### Entity Summary Diagram

```
┌──────────┐     ┌──────────────┐     ┌─────────────────┐
│  Doctor   │────►│CalibrationJob│────►│ PublishedModel   │
│           │     │              │  1:1│                   │
│ role:     │     │ status       │     │ name, desc        │
│ CLINICIAN │     │ alpha        │     │ labels_json       │
│ DEVELOPER │     │ result_json  │     │ alpha, lamhat     │
└──────┬───┘     │ verdict (NEW)│     │ artifact_path     │
       │         │ is_published │     │ visibility        │
       │         └──────────────┘     │ config_json       │
       │                              │ validation_verdict│
       │                              │ is_active         │
       ▼                              └────────┬──────────┘
┌──────────┐                                   │
│ Patient  │                                   │
└──────┬───┘                                   │
       │                                       │
       ▼                                       ▼
┌──────────────────────────────────────────────────┐
│  Prediction                                       │
│  patient_id, doctor_id, image_path                │
│  findings_json, alpha, lamhat                     │
│  published_model_id (NEW, nullable FK)            │
└──────────────────────────────────────────────────┘
```

---

## Section 4 — Release and Visibility Logic

### Visibility Enum

```python
class ModelVisibility(str, Enum):
    PRIVATE = "private"                          # Only the developer can see it
    CLINICIAN = "clinician"                      # Visible to clinicians for inference
    COMMUNITY = "community"                      # Visible in developer model library
    CLINICIAN_AND_COMMUNITY = "clinician_and_community"  # Both
```

### Visibility Matrix

| Visibility | Developer sees in "My Models" | Clinician sees in model dropdown | Community sees in Model Library |
|------------|------------------------------|----------------------------------|-------------------------------|
| PRIVATE | Yes | No | No |
| CLINICIAN | Yes | Yes (if is_active) | No |
| COMMUNITY | Yes | No | Yes (if is_active) |
| CLINICIAN_AND_COMMUNITY | Yes | Yes (if is_active) | Yes (if is_active) |

### Query Filters

```
Clinician model dropdown:
  WHERE visibility IN ('clinician', 'clinician_and_community')
  AND is_active = True

Community model library:
  WHERE visibility IN ('community', 'clinician_and_community')
  AND is_active = True

Developer's own models:
  WHERE developer_id = current_user.id
```

### Consent Flow

When a developer chooses any visibility **other than PRIVATE**, the system must:

1. **Show a consent dialog** explaining:
   - What data will be shared (model artifact, metadata, calibration parameters)
   - Who can access it (clinicians, developers, or both)
   - That attribution will be maintained (their name/reference on the model)
   - That they can revoke access later by changing visibility back to PRIVATE
   - (For clinician release): That the model will be used for patient diagnosis and the developer assumes responsibility for the model's quality within the scope of their validation
   - Disclaimer that SafeDx provides the platform but clinical deployment in a real setting would require formal regulatory review (future work)

2. **Require explicit confirmation**: The developer must check a checkbox and click "Confirm Release"

3. **Record the consent**:
   - `consent_given_at` = current timestamp
   - `consent_text_hash` = SHA-256 hash of the consent text shown (for audit trail)

4. **Allow later changes**:
   - Developer can change visibility at any time
   - Changing from non-private to private does NOT delete the consent record (for audit)
   - Changing from private to non-private requires re-consent if consent text version has changed

### Visibility Change Rules

| From | To | Requires |
|------|----|----------|
| PRIVATE | Any non-private | Consent dialog |
| CLINICIAN | COMMUNITY | Consent dialog (different audience) |
| COMMUNITY | CLINICIAN | Consent dialog (different audience) |
| Any | PRIVATE | No confirmation needed (restricting access) |
| CLINICIAN | CLINICIAN_AND_COMMUNITY | Consent dialog (expanding audience) |
| CLINICIAN_AND_COMMUNITY | CLINICIAN | No confirmation needed (restricting) |

### Deactivation

- A developer can **deactivate** (`is_active = False`) their published model at any time
- Deactivated models disappear from clinician dropdown and community library
- Existing predictions that used this model are NOT affected (they are historical records)
- The developer can **reactivate** later

---

## Section 5 — Developer-Side UX / Pages

### Sidebar Navigation (Updated)

```
SafeDx Developer
─────────────────
📖  How to Calibrate      → /developer/how-to-calibrate
⚙️  Calibrate Your Model  → /developer/calibrate
✓  Validate Calibration   → /developer/validate
📦  Calibrated Models      → /developer/models        ← NEW
─────────────────
🚪  Logout
```

### Page 1: How to Calibrate Your Model (EXISTING — minor update)

- Add a brief section explaining the publish/release flow
- Mention that after validation, models can be published for clinician or community use
- Describe the label schema requirement (labels.csv column names become the model's labels)

### Page 2: Calibrate Your Model (EXISTING — minor updates)

**Current behavior preserved.** Developer uploads model + dataset + config + alpha → starts calibration job → sees job table.

**Additions:**
- In the jobs table, add a "Published" badge/icon column showing whether each job has been published
- After a job completes (status = DONE), show a subtle prompt: "Validate and publish this model →"

### Page 3: Validate Your Calibration (EXISTING — significant additions)

**Current behavior preserved.** Developer selects a completed job → sees FNR chart, set size chart, verdict.

**Additions:**

After the validation results are shown, add a **"Publish This Model"** action section below the charts:

```
┌──────────────────────────────────────────────────────────────┐
│  📦 Publish This Model                                       │
│                                                              │
│  Your calibration passed validation with verdict: GOOD       │
│  You can now publish this as a reusable model package.       │
│                                                              │
│  [ Publish Model → ]                                         │
└──────────────────────────────────────────────────────────────┘
```

Clicking "Publish Model" opens a **Publish Model Dialog** (modal or dedicated section):

#### Publish Model Form

```
┌──────────────────────────────────────────────────────────────┐
│  Publish Model                                               │
│                                                              │
│  Model Name *          [ CheXpert Chest X-Ray Classifier   ] │
│  Description *         [ Multi-label chest X-ray classifier ] │
│                        [ for 5 common thoracic conditions   ] │
│  Version *             [ 1.0.0                              ] │
│  Modality *            [ Chest X-Ray         ▼ ]             │
│  Intended Use *        [ Screening aid for thoracic         ] │
│                        [ conditions in adult patients       ] │
│                                                              │
│  Labels (from calibration dataset):                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ 1. Cardiomegaly                                      │    │
│  │ 2. Edema                                             │    │
│  │ 3. Consolidation                                     │    │
│  │ 4. Atelectasis                                       │    │
│  │ 5. Pleural Effusion                                  │    │
│  └──────────────────────────────────────────────────────┘    │
│  (These are read from your calibration dataset columns)      │
│                                                              │
│  Auto-filled from calibration:                               │
│  Alpha: 0.10  │  Lamhat: 0.4321  │  Verdict: Good           │
│                                                              │
│  ─── Release Options ───                                     │
│                                                              │
│  Who should have access to this model?                       │
│                                                              │
│  ○ Keep Private (only you can see it)                        │
│  ○ Release for Clinician Use                                 │
│  ○ Release to Developer Community                            │
│  ○ Release for Both Clinician and Community Use              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ ⚠ Release Consent                                    │    │
│  │                                                      │    │
│  │ By releasing this model, you acknowledge that:       │    │
│  │                                                      │    │
│  │ • Your model artifact, metadata, and calibration     │    │
│  │   parameters will be shared with the selected        │    │
│  │   audience                                           │    │
│  │ • For clinician release: this model may be used      │    │
│  │   as a diagnostic aid for patients                   │    │
│  │ • You are responsible for the quality and validity   │    │
│  │   of your model within the scope of your validation  │    │
│  │ • Your name will be displayed as the model creator   │    │
│  │ • You can change visibility or deactivate the model  │    │
│  │   at any time                                        │    │
│  │ • Real clinical deployment requires formal review    │    │
│  │   beyond this platform's validation (see disclaimer) │    │
│  │                                                      │    │
│  │ [ ✓ ] I understand and agree to these terms          │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  [ Cancel ]                        [ Publish Model ]         │
└──────────────────────────────────────────────────────────────┘
```

**Rules:**
- "Publish Model" button is disabled until consent checkbox is checked (if non-private)
- If visibility is PRIVATE, consent box is hidden (no consent needed)
- Labels are auto-populated from the calibration dataset's column names (extracted during calibration)
- If verdict is "unreliable", the publish section shows a warning and the publish button is disabled

### Page 4: Calibrated Models (NEW — Model Library)

**Purpose**: A browsable library of all models released to the developer community.

**URL**: `/developer/models`

**Layout**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Calibrated Models                                                       │
│                                                                          │
│  Browse validated, calibrated models shared by the SafeDx community.     │
│                                                                          │
│  [ Search by name or description...          ]  [ Modality ▼ ] [Verdict▼]│
│                                                                          │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐       │
│  │ CheXpert CXR Classifier    │  │ DermNet Skin Lesion Model   │       │
│  │ v1.0.0                     │  │ v2.1.0                      │       │
│  │                            │  │                             │       │
│  │ Multi-label chest X-ray    │  │ 7-class skin lesion         │       │
│  │ classifier for thoracic    │  │ classifier for dermoscopy   │       │
│  │ conditions                 │  │ images                      │       │
│  │                            │  │                             │       │
│  │ Modality: Chest X-Ray     │  │ Modality: Dermatoscopy     │       │
│  │ Labels: 5 conditions      │  │ Labels: 7 conditions       │       │
│  │ Alpha: 0.10               │  │ Alpha: 0.05               │       │
│  │ Lamhat: 0.4321            │  │ Lamhat: 0.3210            │       │
│  │ Verdict: ● Good           │  │ Verdict: ● Good           │       │
│  │ By: Dr. A. Researcher     │  │ By: ML Lab Team           │       │
│  │ Published: 2026-03-10     │  │ Published: 2026-03-01     │       │
│  │                            │  │                             │       │
│  │ [View Details] [Download]  │  │ [View Details] [Download]  │       │
│  └─────────────────────────────┘  └─────────────────────────────┘       │
│                                                                          │
│  ... more model cards ...                                                │
└──────────────────────────────────────────────────────────────────────────┘
```

**Model Detail Expandable/Modal**:

When "View Details" is clicked, show:
- Full description
- Intended use
- Complete label list
- Alpha, lamhat, coverage
- Validation metrics (FNR, avg set size, n_samples, n_positive)
- Verdict with explanation
- Artifact type
- Creator name and publish date
- Version

**Download**: Allows downloading the model artifact (`.pth`/`.pt` file) for the developer's own use.

**Filtering**:
- Text search (name, description)
- Modality dropdown
- Verdict filter (Good / Review)
- Sort by: newest, alphabetical

---

## Section 6 — Clinician-Side UX / Pages

### Updated Clinician Flow

```
1. Login as clinician
2. Go to Dashboard (new patient diagnosis)
3. ┌─────────────────────────────────┐
   │  Step 1: Select Diagnostic Model │  ← NEW
   │  [ Select a model...        ▼ ] │
   │                                  │
   │  ┌── Model Info Card ────────┐  │  ← NEW
   │  │ CheXpert CXR Classifier  │  │
   │  │ v1.0.0                   │  │
   │  │ 5-label chest X-ray AI   │  │
   │  │ Alpha: 0.10 | Coverage:  │  │
   │  │ 90% | Verdict: Good      │  │
   │  └──────────────────────────┘  │
   └─────────────────────────────────┘
4. Step 2: Enter Patient Info (existing)
5. Step 3: Upload CXR Image (existing)
6. Click "Analyze" → runs inference using SELECTED model
7. Results show THAT model's labels dynamically
```

### Model Selector Component

Located at the top of the DiagnosticDashboard, before the patient form:

```
┌──────────────────────────────────────────────────────────────┐
│  Select Diagnostic Model                                      │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  ▼  CheXpert Chest X-Ray Classifier (v1.0.0)          │  │
│  │     ──────────────────────────────────────────         │  │
│  │     CheXpert Chest X-Ray Classifier (v1.0.0)          │  │
│  │     DermNet Skin Lesion Model (v2.1.0)                 │  │
│  │     Retinal OCT Pathology Detector (v1.2.0)            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌── About This Model ──────────────────────────────────┐   │
│  │                                                      │   │
│  │  CheXpert Chest X-Ray Classifier                     │   │
│  │  Version 1.0.0 · by Dr. A. Researcher                │   │
│  │                                                      │   │
│  │  Multi-label chest X-ray classifier for 5 common     │   │
│  │  thoracic conditions. Designed for screening aid.     │   │
│  │                                                      │   │
│  │  Modality: Chest X-Ray                               │   │
│  │  Labels: Cardiomegaly, Edema, Consolidation,         │   │
│  │          Atelectasis, Pleural Effusion                │   │
│  │  Coverage: 90% (alpha = 0.10)                        │   │
│  │  Validation: ● Good                                  │   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Dynamic Results Table

Currently, the findings table hardcodes 5 rows for 5 diseases. This must become dynamic:

**Before (hardcoded)**:
```
| Finding           | Probability | Uncertainty | In Set |
|-------------------|-------------|-------------|--------|
| Cardiomegaly      | 0.82        | Low         | Yes    |
| Edema             | 0.45        | Medium      | Yes    |
| Consolidation     | 0.12        | High        | No     |
| Atelectasis       | 0.08        | High        | No     |
| Pleural Effusion  | 0.65        | Medium      | Yes    |
```

**After (dynamic from model's labels)**:
```
| Finding           | Probability | Uncertainty | In Set |
|-------------------|-------------|-------------|--------|
| {label_1}         | ...         | ...         | ...    |
| {label_2}         | ...         | ...         | ...    |
| ...               | ...         | ...         | ...    |
| {label_n}         | ...         | ...         | ...    |
```

The findings array returned by the API already includes the label names — the UI just needs to render whatever labels come back instead of assuming 5 fixed ones.

### Updated Dashboard State Flow

```
No model selected
  → Show model dropdown, disable patient form + upload
  → Prompt: "Please select a diagnostic model to begin"

Model selected
  → Show model info card
  → Enable patient form + upload
  → "Analyze" button sends model_id with the request

Inference complete
  → Show results using labels from the selected model
  → Show alpha, lamhat, coverage from the selected model
  → Technical details section shows which model was used
```

### Prediction Detail Page (Updated)

When viewing a saved prediction, also show:
- Which model was used (name, version)
- That model's alpha, lamhat, coverage
- The dynamic labels from that model

For historical predictions (before this feature), show: "Legacy model (pre-platform)" or similar.

### Home Page (Existing — minor update)

- No major changes needed
- Optionally: add a "Model" column to the patient list showing which model was used for the latest prediction

---

## Section 7 — Backend / API Plan

### New Endpoints

#### Published Model Endpoints

| Method | Endpoint | Auth | Role | Purpose |
|--------|----------|------|------|---------|
| POST | `/models/publish` | Yes | Developer | Publish a calibration job as a model package |
| GET | `/models/clinician` | Yes | Clinician | List models released for clinician use |
| GET | `/models/community` | Yes | Developer | List models released for community use |
| GET | `/models/mine` | Yes | Developer | List developer's own published models |
| GET | `/models/{id}` | Yes | Any | Get published model details |
| PATCH | `/models/{id}/visibility` | Yes | Developer (owner) | Change visibility |
| PATCH | `/models/{id}/active` | Yes | Developer (owner) | Activate/deactivate |
| GET | `/models/{id}/download` | Yes | Developer | Download model artifact (community models) |

#### Modified Endpoints

| Method | Endpoint | Change |
|--------|----------|--------|
| POST | `/predict` | Add `model_id` parameter; use that model's artifact + config + lamhat for inference |
| GET | `/predictions/{id}` | Include `published_model` info in response |
| GET | `/history` | Include `published_model` summary in each prediction |

### API Specifications

#### POST `/models/publish`

**Purpose**: Create a Published Model Package from a completed, validated calibration job.

**Request Body** (multipart/form or JSON):
```json
{
  "calibration_job_id": "uuid",
  "name": "CheXpert Chest X-Ray Classifier",
  "description": "Multi-label chest X-ray classifier...",
  "version": "1.0.0",
  "modality": "Chest X-Ray",
  "intended_use": "Screening aid for thoracic conditions...",
  "labels": ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"],
  "visibility": "clinician",
  "consent_agreed": true
}
```

**Validation rules**:
- Job must exist and belong to the current developer
- Job status must be DONE
- Job must not already be published (is_published = False)
- Validation verdict must be "good" or "review" (not "unreliable")
- If visibility != "private", consent_agreed must be True
- All required fields must be present

**Response**: Created PublishedModel object

**Backend logic**:
1. Validate all rules above
2. Copy the model artifact to a dedicated published models storage location (or reference the existing path)
3. Create PublishedModel record with all fields
4. Set CalibrationJob.is_published = True
5. Return the created model

---

#### GET `/models/clinician`

**Purpose**: List all models available for clinician inference.

**Query params**: None (could add pagination later)

**Response**:
```json
{
  "models": [
    {
      "id": "uuid",
      "name": "CheXpert Chest X-Ray Classifier",
      "description": "Multi-label chest X-ray classifier...",
      "version": "1.0.0",
      "modality": "Chest X-Ray",
      "labels": ["Cardiomegaly", "Edema", ...],
      "num_labels": 5,
      "alpha": 0.10,
      "lamhat": 0.4321,
      "coverage": "90%",
      "validation_verdict": "good",
      "intended_use": "Screening aid...",
      "developer_name": "Dr. A. Researcher",
      "created_at": "2026-03-10T..."
    }
  ]
}
```

**Filter**: `WHERE visibility IN ('clinician', 'clinician_and_community') AND is_active = True`

---

#### GET `/models/community`

**Purpose**: List all models available in the developer model library.

**Query params**: `?search=`, `?modality=`, `?verdict=`, `?sort=`

**Response**: Same structure as clinician list but with additional fields:
- `artifact_type`
- `validation_metrics` (FNR, avg_set_size, n_samples, n_positive)

**Filter**: `WHERE visibility IN ('community', 'clinician_and_community') AND is_active = True`

---

#### GET `/models/{id}`

**Purpose**: Get full details of a published model.

**Access control**:
- Developer can always see their own models
- Clinician can see models with clinician visibility
- Developer can see models with community visibility
- Others get 403

**Response**: Full PublishedModel object including all metadata, calibration params, validation metrics.

---

#### POST `/predict` (MODIFIED)

**Current**: Uses hardcoded model + lamhat from startup

**New**: Accept `model_id` parameter

```
POST /predict
Content-Type: multipart/form-data

file: <image>
patient_id: 123
model_id: "uuid"      ← NEW (optional for backward compat, required going forward)
```

**Backend logic**:
1. If `model_id` is provided:
   - Load the published model's artifact (lazy-load + cache)
   - Use that model's config for preprocessing
   - Use that model's lamhat for thresholding
   - Use that model's labels for findings
2. If `model_id` is NOT provided (backward compat):
   - Use the legacy hardcoded model (existing behavior)
3. Save prediction with `published_model_id` FK

---

#### PATCH `/models/{id}/visibility`

**Purpose**: Change a model's visibility.

**Request**:
```json
{
  "visibility": "clinician_and_community",
  "consent_agreed": true
}
```

**Rules**:
- Only the owning developer can change visibility
- Expanding visibility requires consent_agreed = True
- Restricting visibility does not require consent

---

### New Backend Service: `published_model_service.py`

**Responsibilities**:
- `publish_model(job_id, metadata, visibility, consent)` → Create PublishedModel
- `list_clinician_models()` → Query with clinician visibility filter
- `list_community_models(search, modality, verdict, sort)` → Query with community filter
- `list_my_models(developer_id)` → Developer's own models
- `get_model_detail(model_id, requesting_user)` → With access control
- `update_visibility(model_id, new_visibility, consent)` → Change visibility
- `toggle_active(model_id, is_active)` → Activate/deactivate

### Modified Backend Service: `ml_service.py`

**Current**: Loads one model at startup, holds it in memory.

**New**: Model loading becomes dynamic per published model.

**Approach: Lazy Loading with LRU Cache**

```
ModelCache (in-memory, bounded LRU)
  key: published_model_id
  value: (loaded_model, config, lamhat, labels)
  max_size: configurable (e.g., 5 models in memory)
  eviction: least recently used
```

**Why LRU cache, not load-all-at-startup**:
- Models are large (hundreds of MB)
- Loading all published models into memory is impractical
- Most clinicians will use 1-3 models regularly
- LRU keeps hot models in memory, evicts cold ones

**Inference flow**:
```
1. Receive (image_bytes, published_model_id)
2. Check cache for model_id
3. If miss: load artifact from disk, parse config, store in cache
4. Preprocess image using model's config
5. Run forward pass
6. Apply model's lamhat for thresholding
7. Map probabilities to model's labels
8. Return findings
```

### Label Extraction During Calibration

**Current**: Labels are hardcoded as `DISEASES` constant.

**Proposed**: During calibration, extract label names from the `labels.csv` column headers.

The calibration dataset already has a `labels.csv` with columns like:
```
filename, Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion
```

The column names (excluding "filename") become the model's label schema. These should be:
1. Saved as part of the CalibrationJob result (e.g., in result_json)
2. Pre-populated in the publish form
3. Stored in PublishedModel.labels_json

This means **labels are automatically derived from the developer's own data**, not manually entered.

---

## Section 8 — Frontend / Component Plan

### New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ModelSelector` | `clinician/components/ModelSelector.tsx` | Dropdown + info card for selecting a published model |
| `ModelInfoCard` | `shared/components/ModelInfoCard.tsx` | Compact card showing model name, version, labels, alpha, verdict |
| `DynamicFindingsTable` | `clinician/components/DynamicFindingsTable.tsx` | Renders findings table from any label set (replaces hardcoded table) |
| `ModelLibraryPage` | `developer/ModelLibraryPage.tsx` | Browsable grid of community models |
| `ModelCard` | `developer/components/ModelCard.tsx` | Card for a single model in the library |
| `ModelDetailModal` | `developer/components/ModelDetailModal.tsx` | Expanded view of model details |
| `PublishModelDialog` | `developer/components/PublishModelDialog.tsx` | Form + consent flow for publishing |
| `ConsentCheckbox` | `shared/components/ConsentCheckbox.tsx` | Reusable consent display + checkbox |
| `VisibilityBadge` | `shared/components/VisibilityBadge.tsx` | Small badge showing model visibility status |
| `VerdictBadge` | `shared/components/VerdictBadge.tsx` | Color-coded validation verdict indicator |

### Modified Components

| Component | Changes |
|-----------|---------|
| `DiagnosticDashboard` | Add ModelSelector at top; pass model_id to predict API call; use dynamic labels in results |
| `PredictionDetail` | Show which model was used; render dynamic labels |
| `DeveloperDashboard` | Add "Published" badge column; add "Publish" link for completed jobs |
| `ValidateCalibrationPage` | Add publish action section below charts |
| `DeveloperLayout` | Add "Calibrated Models" nav item |
| `HomePage` | Optionally show model name in patient list |

### New API Client Functions

Add to `developerApi.ts`:
```
publishModel(data, token) → PublishedModel
listMyModels(token) → PublishedModel[]
listCommunityModels(params, token) → PublishedModel[]
updateModelVisibility(modelId, visibility, consent, token) → void
toggleModelActive(modelId, isActive, token) → void
downloadModelArtifact(modelId, token) → Blob
```

Add to `clinicianApi.ts`:
```
listClinicianModels(token) → PublishedModelSummary[]
getModelDetail(modelId, token) → PublishedModel
```

Modify in `clinicianApi.ts`:
```
predictImage(file, patientId, modelId, token) → PredictionResponse  // added modelId param
```

### Routing Updates

```typescript
// New developer route
"/developer/models"  →  ModelLibraryPage

// Existing routes — no URL changes, just component updates
"/dashboard"         →  DiagnosticDashboard (updated)
"/predictions/:id"   →  PredictionDetail (updated)
"/developer/validate" →  ValidateCalibrationPage (updated)
"/developer/calibrate" → DeveloperDashboard (updated)
```

---

## Section 9 — Recommended Implementation Phases

### Phase 1: Data Foundation
**Goal**: Create the PublishedModel entity and migration without breaking anything.

**Tasks**:
1. Create the `ModelVisibility` enum
2. Create the `PublishedModel` SQLAlchemy model with all fields
3. Add `validation_verdict` and `is_published` columns to `CalibrationJob`
4. Add `published_model_id` nullable FK to `Prediction`
5. Run database migration
6. Persist validation verdict when validation is computed (update `calibration_service.py`)
7. Extract and persist label names from `labels.csv` during calibration (store in `result_json`)

**Verification**: Existing flows still work. New columns are nullable / have defaults. No frontend changes yet.

---

### Phase 2: Publish Flow (Backend)
**Goal**: Developers can publish calibration jobs as model packages via API.

**Tasks**:
1. Create `published_model_service.py` with `publish_model()` function
2. Create `POST /models/publish` endpoint
3. Create `GET /models/mine` endpoint
4. Create `PATCH /models/{id}/visibility` endpoint
5. Create `PATCH /models/{id}/active` endpoint
6. Implement validation rules (verdict gate, consent requirement, duplicate prevention)
7. Copy/reference model artifact to published models storage
8. Write tests for publish flow

**Verification**: Can publish a model via API. Can list own models. Can change visibility.

---

### Phase 3: Publish Flow (Frontend)
**Goal**: Developers can publish models through the UI.

**Tasks**:
1. Build `PublishModelDialog` component (form + consent)
2. Build `VisibilityBadge` and `VerdictBadge` shared components
3. Add publish action to `ValidateCalibrationPage`
4. Add "Published" badge to jobs table in `DeveloperDashboard`
5. Add API client functions for publishing
6. Wire up the full publish flow

**Verification**: Developer can validate → publish → see published badge. Consent flow works.

---

### Phase 4: Model Library (Backend + Frontend)
**Goal**: The "Calibrated Models" page works for the developer community.

**Tasks**:
1. Create `GET /models/community` endpoint with search/filter
2. Create `GET /models/{id}` endpoint with access control
3. Create `GET /models/{id}/download` endpoint
4. Build `ModelLibraryPage` page
5. Build `ModelCard` component
6. Build `ModelDetailModal` component
7. Add "Calibrated Models" to developer sidebar
8. Implement search, modality filter, verdict filter, sort

**Verification**: Developer can browse community models, view details, download artifacts.

---

### Phase 5: Dynamic Inference (Backend)
**Goal**: The inference service can run any published model, not just the hardcoded one.

**Tasks**:
1. Refactor `ml_service.py` to support dynamic model loading
2. Implement LRU model cache (bounded, configurable)
3. Create `load_published_model(model_id)` function that loads artifact + config + lamhat + labels
4. Modify `run_inference()` to accept a published_model_id and use that model's parameters
5. Update preprocessing to use per-model config (width, height, normalization)
6. Update thresholding to use per-model lamhat
7. Update findings to use per-model labels
8. Create `GET /models/clinician` endpoint
9. Modify `POST /predict` to accept and require `model_id`
10. Save `published_model_id` on Prediction records
11. Maintain backward compatibility: if no model_id, use legacy model

**Verification**: Can call `/predict` with a model_id and get inference using that model's artifact, config, lamhat, and labels.

---

### Phase 6: Clinician Model Selection (Frontend)
**Goal**: Clinicians can select a model and get dynamic results.

**Tasks**:
1. Build `ModelSelector` component (dropdown + info card)
2. Build `ModelInfoCard` component
3. Build `DynamicFindingsTable` component
4. Integrate `ModelSelector` into `DiagnosticDashboard`
5. Update predict API call to include `model_id`
6. Update results rendering to use dynamic labels from the response
7. Update `PredictionDetail` to show which model was used
8. Remove hardcoded disease references from clinician frontend
9. Fetch clinician models on page load

**Verification**: Clinician can select model → upload image → see results with that model's labels. Existing predictions still display correctly.

---

### Phase 7: Polish and Edge Cases
**Goal**: Handle edge cases, improve UX, clean up.

**Tasks**:
1. Handle "no models available" state for clinicians (empty dropdown message)
2. Handle model deactivation (what happens if a model is deactivated while a clinician has it selected)
3. Add loading states for model loading (first-time load may be slow)
4. Add error handling for model loading failures
5. Validate that uploaded model's output dimension matches the number of labels
6. Add "Legacy Model" display for historical predictions without published_model_id
7. Update the "How to Calibrate" guide page
8. End-to-end testing of the full flow
9. Clean up any remaining hardcoded disease references in the codebase

**Verification**: All edge cases handled. Full flow works end to end.

---

### Phase Summary

| Phase | Name | Depends on | Estimated Scope |
|-------|------|------------|-----------------|
| 1 | Data Foundation | — | Backend only, DB + service changes |
| 2 | Publish Flow (Backend) | Phase 1 | Backend only, new service + endpoints |
| 3 | Publish Flow (Frontend) | Phase 2 | Frontend, new components + integration |
| 4 | Model Library | Phase 2 | Backend + Frontend, new page |
| 5 | Dynamic Inference | Phase 1, 2 | Backend, major refactor of ml_service |
| 6 | Clinician Model Selection | Phase 5 | Frontend, major refactor of dashboard |
| 7 | Polish & Edge Cases | Phase 1-6 | Both, cleanup and hardening |

---

## Section 10 — Edge Cases / Risks / Constraints

### 1. Incompatible Model Architectures

**Risk**: A developer uploads a `.pth` file that is just state_dict weights (not a self-contained model). Loading it requires knowing the model architecture class.

**Current state**: The calibration service already handles this — it tries `torch.load()` and falls back to TorchScript. But at publish-time inference, the same artifact must be loadable.

**Mitigation**:
- During calibration, if the model loads successfully, it will also load successfully at inference time (same loading code)
- Store the artifact type ("pytorch_full" vs "torchscript") in the PublishedModel
- For the prototype, require that uploaded models are either full `torch.save(model)` or TorchScript
- Document this requirement in the "How to Calibrate" guide
- Future: support ONNX format for architecture-independent inference

### 2. Output Dimension Mismatch

**Risk**: Model has 5 output neurons but developer specifies 7 labels, or vice versa.

**Mitigation**:
- During calibration, the system already runs inference and gets output shape
- Store `num_outputs` from calibration alongside labels
- At publish time, validate that `len(labels) == num_outputs`
- If mismatch, block publishing with a clear error

### 3. Dynamic Label Rendering

**Risk**: Frontend breaks if a model has 1 label, or 50 labels, or labels with special characters.

**Mitigation**:
- Design `DynamicFindingsTable` to handle any number of labels gracefully
- Set reasonable limits: minimum 2 labels, maximum 50 labels (validated at publish time)
- Sanitize label names (strip leading/trailing whitespace, reject empty strings)
- Test with various label counts (2, 5, 10, 20)

### 4. Missing or Corrupt Config

**Risk**: Published model has no config, or config is missing required fields (width, height, normalization params).

**Mitigation**:
- Define a default config (224x224, standard normalization) as fallback
- At publish time, validate that config has all required fields
- Store validated config in PublishedModel, not just a reference to a file

### 5. Invalid Release State

**Risk**: Developer publishes a model, then the underlying calibration job is deleted, or the artifact file is moved/deleted.

**Mitigation**:
- When a job is published, set `is_published = True` and prevent deletion of that job
- Copy (don't just reference) the model artifact to a dedicated published models directory
- At inference time, gracefully handle missing artifact: return error "Model temporarily unavailable"

### 6. Model Cache Memory Pressure

**Risk**: Multiple large models loaded simultaneously exhaust server memory.

**Mitigation**:
- LRU cache with configurable max size (default: 3 models)
- Log cache hits/misses for monitoring
- Return clear error if model cannot be loaded (e.g., OOM)
- Future: move to a model serving architecture (TorchServe, Triton) for production

### 7. Clinician Confusion from Too Many Models

**Risk**: If 50 models are released, the clinician dropdown becomes overwhelming.

**Mitigation**:
- Group models by modality in the dropdown
- Show clear info cards so clinicians can make informed choices
- Consider "featured" or "recommended" flags in future
- For the prototype, this is acceptable — the population of models will be small

### 8. Stale Predictions After Model Deactivation

**Risk**: A model is deactivated, but clinicians have saved predictions that reference it.

**Mitigation**:
- Predictions are historical records — they should remain viewable even if the model is deactivated
- In PredictionDetail, if the model is deactivated, show: "Model: CheXpert v1.0.0 (no longer active)"
- The prediction data (findings, probabilities) is already stored on the Prediction record, so it doesn't depend on the model being loadable

### 9. Backward Compatibility for Existing Predictions

**Risk**: Existing predictions have no `published_model_id` — they were made with the hardcoded model.

**Mitigation**:
- `published_model_id` is nullable on Prediction
- Frontend checks: if `published_model_id` is null, show "Legacy Model" or "SafeDx Default Model"
- No data migration needed for existing predictions

### 10. Concurrent Calibration/Publishing Race Condition

**Risk**: Developer starts publishing while calibration job is still running, or publishes the same job twice from two browser tabs.

**Mitigation**:
- Publish endpoint checks `job.status == DONE` (rejects if not)
- Database unique constraint on `PublishedModel.calibration_job_id` prevents double-publish
- Set `is_published = True` atomically with creating the PublishedModel (same transaction)

### 11. Large File Downloads in Model Library

**Risk**: Model artifacts can be 500MB+. Downloading them through the API may timeout or fail.

**Mitigation**:
- Use streaming responses for artifact downloads
- Show file size in the model detail card so developers know what to expect
- Future: use Supabase Storage or S3 presigned URLs for direct downloads

### 12. Consent Text Changes

**Risk**: Consent text is updated after a developer has already agreed. Does their consent still apply?

**Mitigation**:
- Store `consent_text_hash` on PublishedModel — the hash of the exact text they agreed to
- If consent text changes, existing consents remain valid (they agreed to an earlier version)
- Only new publications or visibility expansions require agreeing to the current text

---

## Section 11 — Future Work

### 1. Human Approval / Clinical Review Workflow

The current prototype uses automated validation (FNR analysis, verdict assignment) as the gate for clinician release. In a real-world deployment:

- A **clinical review board** or **domain expert panel** should review models before they can be released for patient use
- Approval states: `PENDING_REVIEW → UNDER_REVIEW → APPROVED / REJECTED`
- Reviewers should see validation metrics, intended use, sample predictions, and model documentation
- Rejection should include feedback so the developer can improve and resubmit

### 2. AI/ML Expert Review

Beyond clinical review, an ML expert review could assess:
- Model architecture appropriateness
- Training data quality and representativeness
- Potential biases (demographic, acquisition-device, institution)
- Calibration quality beyond automated metrics

### 3. Licensing and Governance

- Model licensing: developers should choose a license (e.g., CC-BY, MIT, proprietary)
- Usage terms: can a model be used commercially? Modified? Redistributed?
- Data governance: what training data was used? Any privacy concerns?
- Regulatory compliance: FDA/CE marking considerations for clinical deployment

### 4. Model Versioning Improvements

Current: simple version string (e.g., "1.0.0")
Future:
- Version lineage: v1.0.0 → v1.1.0 → v2.0.0
- Deprecation: mark older versions as deprecated, suggest upgrade
- Side-by-side comparison of versions
- Automatic migration of clinician model selection when a new version is published

### 5. Audit Trail

- Log every inference run: who, when, which model, which patient
- Log every model state change: published, visibility changed, deactivated
- Immutable audit log for regulatory compliance
- HIPAA/GDPR considerations for patient data

### 6. Model Library Enhancements

- **Ratings/reviews**: developers can rate and review community models
- **Usage statistics**: how many times a model has been used for inference
- **Tags**: searchable tags beyond modality
- **Collections**: curated collections of models for specific use cases
- **Forking**: developers can fork a community model as a starting point

### 7. Advanced Inference Features

- **Batch inference**: upload multiple images at once
- **Comparison mode**: run same image through multiple models, compare results
- **Ensemble mode**: combine predictions from multiple models
- **Longitudinal tracking**: track predictions for the same patient over time

### 8. Multi-Tenancy and Organization Support

- Organizations (hospitals, research labs) as first-class entities
- Organization-level model visibility (shared within an org but not publicly)
- Admin roles within organizations
- Organization-branded instances

### 9. Model Monitoring in Production

- Track prediction distribution drift over time
- Alert if a model's predictions deviate from expected patterns
- Performance dashboards for model owners
- Feedback loop: clinicians can flag incorrect predictions

### 10. Expanded Model Format Support

- ONNX models (architecture-independent)
- TensorFlow/Keras models
- Scikit-learn models (for tabular data)
- Custom inference containers

### 11. API Access for External Systems

- REST API for external systems to submit images and receive predictions
- Webhook notifications for model updates
- Integration with hospital PACS/EHR systems
- HL7 FHIR compatibility

### 12. Internationalization

- Multi-language UI
- Localized label names (e.g., disease names in different languages)
- Region-specific regulatory compliance

---

## Appendix A — Full Entity Field Reference

### PublishedModel — Complete Field List

```
id                      UUID        Primary key
calibration_job_id      UUID FK     → CalibrationJob.id (unique)
developer_id            Integer FK  → Doctor.id

-- Identity
name                    String(150) Display name
description             Text        What the model does
version                 String(20)  Semantic version

-- Classification
modality                String(100) E.g., "Chest X-Ray"
intended_use            Text        Clinical context

-- Technical Package
artifact_path           String      Server path to .pth/.pt file
artifact_type           String(20)  "pytorch" / "torchscript"
config_json             JSON        { width, height, pixel_mean, pixel_std, ... }
labels_json             JSON        ["Cardiomegaly", "Edema", ...]
num_labels              Integer     len(labels_json)

-- Calibration Outputs
alpha                   Float       Selected alpha
lamhat                  Float       Computed threshold
lamhat_result_json      JSON        Full lamhat computation result

-- Validation Outputs
validation_verdict      String      "good" / "review"
validation_metrics_json JSON        { fnr, avg_set_size, n_samples, n_positive }

-- Visibility & Release
visibility              Enum        PRIVATE / CLINICIAN / COMMUNITY / CLINICIAN_AND_COMMUNITY
is_active               Boolean     Default True

-- Consent
consent_given_at        DateTime    When agreed (nullable if PRIVATE)
consent_text_hash       String(64)  SHA-256 of consent text version

-- Timestamps
created_at              DateTime
updated_at              DateTime
```

### CalibrationJob — New Fields

```
validation_verdict      String      "good" / "review" / "unreliable" (nullable)
is_published            Boolean     Default False
```

### Prediction — New Field

```
published_model_id      UUID FK     → PublishedModel.id (nullable)
```

---

## Appendix B — Consent Text Template

```
MODEL RELEASE AGREEMENT

By releasing your model on SafeDx, you acknowledge and agree to the following:

1. SHARING: Your model artifact, calibration parameters (alpha, lamhat),
   validation metrics, and the metadata you provide (name, description,
   labels, intended use) will be shared with the audience you select.

2. CLINICIAN USE (if applicable): Your model may be used as a diagnostic
   aid by clinicians for patient care within the SafeDx platform. You
   affirm that your model has been validated through SafeDx's calibration
   and validation pipeline.

3. COMMUNITY USE (if applicable): Other developers and researchers will
   be able to view your model's details and download your model artifact
   for their own research and development purposes.

4. ATTRIBUTION: Your name will be displayed as the model creator
   alongside the model.

5. RESPONSIBILITY: You are responsible for the quality and accuracy of
   your model within the scope of the validation performed. SafeDx
   provides the platform and validation tools but does not independently
   verify model correctness.

6. REVOCABILITY: You may change your model's visibility or deactivate
   it at any time. Existing predictions made with your model will
   remain in the system as historical records.

7. DISCLAIMER: This platform is designed for research and prototype
   use. Real-world clinical deployment of AI diagnostic models requires
   formal regulatory review and approval, which is outside the scope
   of this platform.
```

---

## Appendix C — API Response Schemas

### PublishedModelSummary (for lists)

```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "version": "string",
  "modality": "string",
  "num_labels": 5,
  "labels": ["string"],
  "alpha": 0.10,
  "lamhat": 0.4321,
  "coverage": "90%",
  "validation_verdict": "good",
  "developer_name": "string",
  "created_at": "datetime",
  "is_active": true
}
```

### PublishedModelDetail (for single model view)

```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "version": "string",
  "modality": "string",
  "intended_use": "string",
  "num_labels": 5,
  "labels": ["string"],
  "alpha": 0.10,
  "lamhat": 0.4321,
  "coverage": "90%",
  "artifact_type": "pytorch",
  "config": { "width": 224, "height": 224, "pixel_mean": 128.0, "pixel_std": 64.0 },
  "validation_verdict": "good",
  "validation_metrics": {
    "fnr": 0.05,
    "avg_set_size": 2.3,
    "n_samples": 500,
    "n_positive": 320
  },
  "developer_name": "string",
  "developer_id": 1,
  "visibility": "clinician_and_community",
  "is_active": true,
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### PredictionResponse (updated)

```json
{
  "id": 1,
  "patient_id": 123,
  "image_path": "url",
  "top_finding": "string",
  "top_probability": 0.82,
  "prediction_set_size": 3,
  "coverage": "90%",
  "alpha": 0.10,
  "lamhat": 0.4321,
  "findings": [
    {
      "finding": "string (from model's labels)",
      "probability": 0.82,
      "uncertainty": "Low",
      "in_prediction_set": true
    }
  ],
  "model": {
    "id": "uuid",
    "name": "string",
    "version": "string",
    "modality": "string"
  },
  "created_at": "datetime"
}
```
