# Drop `best.pt` here

Put the SAM-HeLa checkpoint at:

```
models/checkpoints/sam_hela/best.pt
```

Keep the filename `best.pt`. The app finds it automatically on startup.

You can also **just launch the app** — the first time SAM is used it
pops up a dialog asking where the file lives (pick the file directly,
or pick a folder containing it). The choice is remembered across
launches.

Other ways to point at the file:
- **In the app:** I/O → Output settings… → SAM-HeLa checkpoint →
  Local file → Browse…
- **Env var:** `EYE_LABELLER_SAM_HELA_LOCAL_PATH=/full/path/to/best.pt`
