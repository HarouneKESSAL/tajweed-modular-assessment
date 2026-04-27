## Scripts Layout

The flat script layer has been split by responsibility:

- `burst/`: burst-rule training and data builders
- `content/`: content training, evaluation, tuning, and failure analysis
- `data/`: shared manifest/alignment/feature preparation utilities
- `duration/`: duration training, analysis, tuning, and localized/fusion work
- `system/`: end-to-end inference and suite/pipeline evaluation
- `transition/`: transition training, analysis, tuning, and localized work

Root-level `scripts/*.py` files are compatibility wrappers.
They forward to the new module folders so existing commands still work.
