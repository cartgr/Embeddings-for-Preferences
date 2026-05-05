# Datasets

11 human-preference triplet datasets in `processed/eval/`:

| Source | Dataset                                                    | Participants    | Triplets           |
|--------|------------------------------------------------------------|----------------:|-------------------:|
| GSC    | abortion_gen, abortion_val, chatbot_gen                    | 100 / 100 / 100 | 0.8K / 3.5K / 1.1K |
| Polis  | seattle, bowling_green, brexit, canadian, ubi              | 13–222          | 0.5K – 1.1M        |
| Remesh | campus_protests, foreign_intervention, right_to_assemble   | 289–298         | 2K – 25K           |

## Schema

JSONL, one triplet per line:

```json
{"anchor_texts":   ["user's own text 1", "user's own text 2"],
 "preferred":      "higher-rated statement",
 "dispreferred":   "lower-rated statement",
 "dataset":        "gsc_abortion_gen",
 "participant_id": "gen1"}
```

## Other directories

- `processed/issues/` — filtered political issues (input to the synthetic-data pipeline).
- `processed/opinions/` — LLM-generated opinions per issue.
- `processed/triplets/` — synthetic training triplets and hard eval triplets.
- `models/best/` — LoRA adapters, 4 encoder families × 5 seeds.
- `results/` — experiment JSONs and PSD probe weights.

## Licensing

Datasets retain their upstream licenses. The GSC, Polis, Remesh,
Habermas, and Kialo datasets are distributed under their own terms;
consult the original sources before redistributing.
