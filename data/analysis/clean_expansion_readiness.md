# Clean Expansion Readiness

## Clean Pool

- Raw rows: `6828`
- Clean matched rows: `2081`
- Clean unique verse IDs: `28`
- Clean unique surahs: `13`

## Candidate Rule Counts

- Duration rows with duration rules: `1415`
- Duration rule positions: `{'madd': 2035, 'ghunnah': 632}`
- Transition rows by label: `{'none': 1805, 'ikhfa': 227, 'idgham': 49}`
- Transition ambiguous rows skipped by current builder: `0`
- Burst rows by label: `{'none': 1456, 'qalqalah': 625}`

## Official Manifests

| Module | Rows | Unique Verse IDs | Unique Texts | Notes |
| --- | ---: | ---: | ---: | --- |
| duration | `973` | `8` | `8` | `{}` |
| transition | `690` | `27` | `27` | `{'none': 414, 'ikhfa': 227, 'idgham': 49}` |
| burst | `1597` | `29` | `28` | `{'none': 958, 'qalqalah': 639}` |
| content_chunks | `1944` | `8` | `17` | `{}` |

## Recommendation

Decision: `expand_with_gates_not_blind_training`

- Transition and burst already cover most of the clean matched rule pool.
- Content has the largest need for more text diversity, but previous scaled alignment experiments did not beat the tuned chunked baseline.
- Duration is already strong, so expansion should be verse-held-out and promotion-gated before replacing the baseline.

Next experiment:
- Create a stricter content expansion split from clean matched rows.
- Train content with a capped/curriculum mix so diverse aligned chunks do not overpower the stable baseline chunks.
- Compare only against the tuned chunked open baseline.
- Promote only if strict text-held-out exact and char accuracy improve.
