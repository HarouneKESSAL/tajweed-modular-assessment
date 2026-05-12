
## Transition threshold decision

Transition thresholds were ablated against argmax/no-threshold decoding.

Result:

```text
default transition subset:
  with thresholds    = 0.901
  without thresholds = 0.910

gold-only:
  with thresholds    = 0.901
  without thresholds = 0.910

extended:
  with thresholds    = 0.835
  without thresholds = 0.852

Decision:

Default transition decoding = argmax / no thresholds
Legacy threshold behavior remains available with:
--enable-transition-thresholds

