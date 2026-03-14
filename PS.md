# HiLabs Workshop

(You can also participate in groups (maximum 3 participants/per group)

---

# 🧠 Building Evals and Reliability Layers for Generative AI in Healthcare

---

## 📌 Workshop Overview

Healthcare AI systems are increasingly used to process large volumes of medical documentation. These systems typically combine **OCR pipelines**, **clinical NLP models**, and **structured extraction pipelines** to convert unstructured clinical records into structured medical data.

While these systems can extract a wide range of clinical information, ensuring **reliability and trustworthiness** is critical. Errors in entity extraction, temporality reasoning, or subject attribution can lead to incorrect clinical insights.

This workshop focuses on **designing evaluation frameworks and reliability layers** for clinical AI pipelines.

Participants will analyse real-world AI outputs and identify weaknesses across different dimensions of clinical reasoning.

---

## ⚙️ System Overview

Participants will work with outputs generated from a clinical AI pipeline that processes medical charts.

The pipeline consists of the following components:

### 1️⃣ OCR PipeLine

Converts scanned medical charts (PDFs) into structured markdown text.

### 2️⃣ Entity Extraction Pipeline

A clinical NLP system extracts structured medical entities from the OCR output.

Example entity types include:

- Diagnosis
- Procedure
- Lab
- Vital
- Medication
- Family history mentions
- Scheduled future events

Each entity is enriched with additional metadata.

---

## 📦 Dataset Provided

https://drive.google.com/drive/folders/1Elnuj6n7QDazhmSsgCMkL1g9UOBTEdkQ

Dataset includes:

- Structured OCR output
- Extracted clinical entities
- Metadata attributes for each entity

**Use Below Link for getting more information about JSON Schema**

[**Entity JSON Structure**](https://www.notion.so/Entity-JSON-Structure-3e3fe8f867df83e5abe0015cb2c37421?pvs=21)

Use Above link for Accessing the Data

Directory Structure

```jsx
 test_data/
    ├── [Folder_1]/
    │   ├── [Folder_1].json  (modified)
    │   └── [Folder_1].md
    ├── [Folder_2]/
    │   ├── [Folder_2].json  (modified)
    │   └── [Folder_2].md
    └── ... (30 folders)
```

---

## 🎯 Objective of the Exercise

Participants are **NOT required to improve or retrain the model**.

Instead, the goal is to **design an evaluation and reliability framework** that helps answer key questions about the system’s performance.

Participants should also analyze:

- Where the system performs well
- Where the system fails
- Which types of errors occur most frequently
- Which reasoning dimensions are unreliable

---

## 📤 Submission Format

Participants should submit a **GitHub repository** containing the evaluation outputs, report, and analysis script.

**Repository Structure**

```
repo_name/
│
├── output/
│   ├── [file_1].json
│   ├── [file_2].json
│   └── ...
│
├── report.md
└── test.py
```

---

**1️⃣ Output Folder**

Create a folder named **`output`**.

- For every file in `test_data`, generate a **JSON evaluation report**.
- The output file name must be **exactly the same as the input test file name**.

Example:

```
test_data/chart_01.json
output/chart_01.json
```

Each JSON file should contain **evaluation breakdowns for that chart**.

---

**2️⃣ Output JSON Schema**

Each output file should follow this structure:

```json
{
  "file_name": "string",

  "entity_type_error_rate": {
    "MEDICINE": 0.0,
    "PROBLEM": 0.0,
    "PROCEDURE": 0.0,
    "TEST": 0.0,
    "VITAL_NAME": 0.0,
    "IMMUNIZATION": 0.0,
    "MEDICAL_DEVICE": 0.0,
    "MENTAL_STATUS": 0.0,
    "SDOH": 0.0,
    "SOCIAL_HISTORY": 0.0
  },

  "assertion_error_rate": {
    "POSITIVE": 0.0,
    "NEGATIVE": 0.0,
    "UNCERTAIN": 0.0
  },

  "temporality_error_rate": {
    "CURRENT": 0.0,
    "CLINICAL_HISTORY": 0.0,
    "UPCOMING": 0.0,
    "UNCERTAIN": 0.0
  },

  "subject_error_rate": {
    "PATIENT": 0.0,
    "FAMILY_MEMBER": 0.0
  },

  "event_date_accuracy": 0.0,
  "attribute_completeness": 0.0
}
```

All values should be **rates between 0 and 1**.

---

**3️⃣ Report**

Include a **`report.md`** file containing:

- Quantitative evaluation summary
- Error heat-map
- Top systemic weaknesses in the pipeline
- Proposed guardrails for improving reliability

---

**4️⃣ Analysis Script**

Provide an entry-point script named **`test.py`**.

The script should:

- Accept a **JSON input file**
- Perform **entity evaluation and metric computation**
- Generate the **corresponding output JSON report**

Example usage:

```bash
python test.py input.json output.json
```

The script may use **additional helper modules**, but **`test.py` must remain the main entry point** for running the evaluation.

---

**Note** - There are **many possible solutions** for this exercise. Participants are free to use **rule-based methods, statistical analysis, or LLM-based approaches** to analyze the dataset and generate evaluation outputs.

If your solution requires an **LLM**, you may use any provider of your choice. Below are some **recommended free-tier options** that can be used without requiring paid billing for basic experimentation.

**Use Below Link for Free Tier LLM Guide**

[Recommended Free-Tier LLM Options](https://www.notion.so/Recommended-Free-Tier-LLM-Options-532fe8f867df8264a323012cb1b72a4b?pvs=21)

---