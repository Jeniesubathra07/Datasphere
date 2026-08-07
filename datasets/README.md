# Datasets

Place the Child Education Risk Intelligence dataset files here.

## Expected files

Push your worked dataset and notebook artifacts to this folder:

| File | Description |
| --- | --- |
| `raw/student_education_risk.csv` | Main dataset (~200K students, ~47 features) |
| `raw/student_education_risk.zip` | Zipped CSV (auto-extracted on load) |
| `raw/*.csv` | Any additional source tables |

## Feature domains (from project spec)

- Student: age, gender, grade, enrollment
- Academic: attendance, test scores, pass/fail, transfers
- Socioeconomic: income, family size, parent education
- Family: orphan status, single parent, siblings
- School: infrastructure, teacher-student ratio, school type
- Access & community: distance, transport, migration, conflict
- Well-being: health, bullying, child labour, special needs
- Digital access: electricity, internet, device ownership

## Target

Four risk levels: `Low`, `Medium`, `High`, `Critical`

If `risk_level` is missing, the pipeline constructs labels from attendance, academics, and vulnerability indicators.
