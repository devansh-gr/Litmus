# external_test cleaning log

Source: `external_test.jsonl` (300) -> cleaned `external_test_clean.jsonl` (290). Removed 10 entries that are not instances of their label.

| removed text | labeled | why removed |
|---|---|---|
| Which fallacy is used to promote something based on popularity? | social-proof-conformity | quiz question stem, not manipulative text |
| This type of ad taps into a person's desire to be part of a group. | social-proof-conformity | definition of the technique, not an instance |
| This coin has landed heads-up nine times in a row. So it will probably land tail | social-proof-conformity | gambler's fallacy, not appeal-to-popularity |
| President Clinton is an advocate of socialized medicine, which is a form of comm | social-proof-conformity | guilt-by-association smear, not appeal-to-popularity |
| Presenting an unqualified person or institution as a source of credible informat | authority-appeal | definition of the fallacy, not an example of it |
| This type of propaganda implies that since EVERYONE else is buying a product, so | social-proof-conformity | definition of the technique, not an instance |
| Invoke shared values and principles. They call upon the audience’s sense of righ | authority-appeal | definition of a different technique (appeal to shared values), not appeal-to-authority |
| when evidence boils down to "everybody's doing it, so it must be a good thing to | social-proof-conformity | garbled meta-fragment describing the fallacy, not a clean instance |
| Homosexuality is / ought to be morally wrong (moral property) because it is not  | authority-appeal | naturalistic fallacy (appeal to nature), not appeal-to-authority |
| This makes you think you need to believe or buy something because everyone else  | social-proof-conformity | definition + quiz stem, not an instance |

Removal rule: only DEFINITIONS of a technique, QUIZ-QUESTION stems, and clear CROSS-FALLACY mislabels are dropped — all genuine and borderline examples are kept. The original file is preserved for provenance; regenerate with `python tests/clean_external_test.py`.
