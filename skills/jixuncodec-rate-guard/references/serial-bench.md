# Serial benchmark notes

## Known fingerprints

| Rate | `idx:rec` fingerprint |
|---|---|
| 1k | `70cc287327d70e4b:3aa0e5aa76f02e8c` |
| 3k | `73e73efd4dbdc601:10cf457b03bfcf6e` |
| 6k | `6466db7b1b12fa53:069f2b9cf1375390` |

These fingerprints identify the deterministic 1-second codec benchmark output. A changed fingerprint can be intentional after a model or kernel change, but it must be reviewed before treating the build as a regression.

## Serial behavior

- Open the board at 115200 baud with DTR and RTS deasserted.
- Wait for the `nsh>` prompt before sending commands.
- Send `jixun bench 1` once per round and wait for the next prompt.
- Keep USB token output, PCM dump, and beeps disabled during deterministic measurements.

## Runtime skill deployment

The skill folder is self-contained. In an openvela controller or Linux edge gateway, copy it to:

```text
/data/agent/skills/jixuncodec-rate-guard/
```

The board-side firmware remains responsible for codec execution. The runtime skill orchestrates the serial benchmark and applies the consistency gates.
