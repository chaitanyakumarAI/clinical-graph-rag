# Test questions for the sample PDFs

Ingest all three PDFs in this folder first:

- `diabetes_management_guide.pdf` — Type 2 Diabetes drugs (metformin, empagliflozin, semaglutide, glipizide, insulin)
- `hypertension_cardiology_guide.pdf` — blood-pressure drugs, atrial fibrillation, warfarin, apixaban
- `drug_interactions_reference.pdf` — interactions that **link the other two documents**

The documents are synthetic educational text written for testing, not clinical guidance.
Every expected answer below is stated in the PDFs, so if the RAG answers
something different, it's a retrieval or extraction problem.

---

## 1. `vector_only` — definitions and descriptions

| Question | Expected answer | Source |
|---|---|---|
| What is Type 2 Diabetes? | Chronic metabolic disease: insulin resistance plus progressive loss of beta-cell insulin secretion → hyperglycemia | diabetes guide |
| How does empagliflozin lower blood glucose? | Inhibits SGLT2 in the proximal tubule → less glucose reabsorbed, more excreted in urine | diabetes guide |
| Why is hypertension called a silent disease? | It is usually asymptomatic | hypertension guide |
| How is hypoglycemia treated? | 15 g fast-acting carbohydrate, recheck after 15 min; glucagon for severe episodes | diabetes guide |

## 2. `hybrid_single` — one relationship hop

| Question | Expected answer | Graph relation |
|---|---|---|
| What are the side effects of metformin? | Nausea, diarrhea, abdominal discomfort, metallic taste, vitamin B12 deficiency, (rare) lactic acidosis | `CAUSES` |
| What are the symptoms of atrial fibrillation? | Palpitations, shortness of breath, fatigue, dizziness | `HAS_SYMPTOM` |
| What drugs treat hypertension? | Lisinopril, losartan, amlodipine, hydrochlorothiazide, metoprolol | `TREATS` |
| When is semaglutide contraindicated? | Personal/family history of medullary thyroid carcinoma or MEN2 | `CONTRAINDICATED_FOR` |
| What does warfarin inhibit? | Vitamin K epoxide reductase (VKORC1) | `INHIBITS` |
| Why does lisinopril cause a cough? | Bradykinin accumulation | `CAUSES` |

## 3. `hybrid_multi` — answers need facts from more than one document

These are the real test of the graph: the pieces live in **different PDFs**.

| Question | Expected answer | How the hops connect |
|---|---|---|
| Which diabetes drugs interact with blood-pressure medications? | Empagliflozin + hydrochlorothiazide (volume depletion); insulin/glipizide + metoprolol (masked hypoglycemia) | drug class (diabetes guide) → interaction (reference) → drug class (hypertension guide) |
| Why should a patient on insulin and metoprolol monitor glucose more often? | Metoprolol masks tachycardia and tremor, the warning signs of the hypoglycemia that insulin causes; sweating is not masked | insulin → hypoglycemia (diabetes guide); metoprolol masks symptoms (reference) |
| A diabetic patient with albuminuria needs a blood-pressure drug. What is preferred and what interaction should be watched? | Lisinopril or losartan (slow diabetic nephropathy); watch hyperkalemia with spironolactone or potassium supplements; never combine the two | hypertension guide + reference |
| Which drugs protect both the heart and the kidneys in Type 2 Diabetes? | Empagliflozin (heart failure + CKD); lisinopril/losartan (nephropathy); semaglutide (cardiovascular events) | all three documents |
| What can cause acute kidney injury in a patient taking lisinopril and hydrochlorothiazide? | Adding an NSAID such as ibuprofen (the "triple whammy") | reference + hypertension guide |
| Which drug raises warfarin levels by inhibiting the enzyme that metabolizes it? | Amiodarone inhibits CYP2C9 → raised INR, bleeding; reduce warfarin dose 30–50% | warfarin → CYP2C9 (hypertension guide) ← amiodarone (reference) |

## 4. `gds_augmented` — graph algorithms

| Question | What a good answer looks like |
|---|---|
| What connects metformin to lactic acidosis? | Path via renal impairment / chronic kidney disease, or via iodinated contrast → kidney injury |
| What connects hypertension to stroke? | Direct link, plus hypertension → atrial fibrillation → stroke |
| Which drugs are most similar to lisinopril in the knowledge graph? | Losartan (shared: treats hypertension, causes hyperkalemia, contraindicated in pregnancy, interacts with spironolactone) |
| Which entities are most central in the drug interaction network? | Warfarin, lisinopril and hypoglycemia-related drugs should rank high |

## 5. Should say "insufficient information"

These are **not** in the PDFs. A grounded RAG should refuse rather than answer from general knowledge.

| Question | Why |
|---|---|
| What is the dose of atorvastatin for high cholesterol? | Statins are never mentioned |
| Does metformin interact with alcohol? | Not in the documents |
| What are the side effects of apixaban in children? | Pediatric use is not covered |

---

## What to check in the UI

- **Citations:** answers should cite the right filename and page, e.g. `[drug_interactions_reference.pdf, p.1]`.
- **Route:** the route pill should roughly match the section heading above.
- **Linked entities:** cross-document questions should link entities from both documents.
- **Graph Explorer:** after ingesting, filter by `Drug` — `Lisinopril` should connect to both `Hypertension` and `Spironolactone`.
