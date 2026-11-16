# Data guide

All structured files use UTF-8 JSON or JSON Lines (one JSON object per line).

| File | Rows | Purpose |
|---|---:|---|
| `data/questions.jsonl` | 501 | Canonical resource inventory: 423 base-collection tasks plus 78 extension questions |
| `data/reviewed_core.jsonl` | 138 | Reviewed versions used in the human evaluation, including contexts and reference answers |
| `data/broad_questions.jsonl` | 60 | Original question versions and answer specifications used by the automated broad scorers |
| `data/required_rules.jsonl` | 138 | Required atoms, their roles, and AND/OR omission-scoring logic |
| `data/evidence_sets.jsonl` | 138 | Primary and dependency evidence IDs for each reviewed question |
| `data/sources.jsonl` | 81 | 76 primary passages and 5 supplementary records; URLs, text, provenance, and recorded date evidence |

`canonical_question_id` joins the resource, reviewed core, annotations, and result records. Keep the study-specific question version: a shared canonical ID does not imply identical wording or answer requirements. `canonical_source_id` in source records joins the supplied evidence IDs. `fact_group_id` and `dependency_group_id` define the human sensitivity groupings.

The inventory's `historical_ids` preserves the internal `ARR_v1` label for traceability. It is a source-version identifier, not an evaluation method or an assertion of publication in ACL Rolling Review.

In `required_rules.jsonl`, required atoms can express values, eligibility conditions, qualifications, or dependencies. The original logic chooses an admissible alternative path and evaluates all coverage metrics on the same path. Given premises and supplementary propositions are excluded from omission penalties. A `null` role-specific coverage value is structurally inapplicable, not a zero or an unreviewed item.

Source dates are preserved as recorded, including nulls; this export does not fill gaps or infer legal validity. The paper describes its legal reference date. Source records and dates must be read with that scope, rather than treated as a claim that the law remains current.

## Human records

`responses.jsonl` retains the exact answer text, native request body, question ID, model ID, condition, evidence IDs, and a response-text hash. Provider credentials, headers, billing records, and transport metadata are excluded.

`initial_reviews.jsonl` has two rows per request, marked `reviewer_1` and `reviewer_2`. These are anonymous review slots, not persistent person identities. Claim segmentation can differ between slots. Agreement therefore uses correctness, whole-answer fidelity, and matched required atoms, not forced claim-to-claim alignment.

`final_labels.jsonl` preserves released semantic annotations and whether they came from agreement or adjudication. `answer_items.jsonl` contains protocol-adjusted metrics. Raw final labels and protocol-adjusted success rates are distinct: failures remain in the denominator, and certain failure statuses receive zero even when a semantic subjudgment is positive.
