# Architecture notes

- `content/`: phoneme-level content verification.
- `duration/`: MFCC + BiLSTM baseline for Madd and Ghunnah.
- `transition/`: hybrid MFCC + SSL-style module for Idgham and Ikhfa'.
- `burst/`: CNN module for Qalqalah-like burst events.
- `fusion/`: diagnosis aggregation and feedback.
