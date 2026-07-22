# Drop `best.pt` here

Put the SAM-HeLa checkpoint at:

```
models/checkpoints/sam_hela/best.pt
```

Keep the filename `best.pt`. On launch the app finds it here and
registers it in the model registry as **sam_hela** automatically — no
configuration needed.

Other ways to point at the file:

- **In the app (recommended):** **Model → Add model…** — give it a
  tag, pick the base architecture (`vit_b` for the standard SAM-HeLa
  fine-tune), and browse to the file. Manage everything later under
  **Settings → SAM Model** (add / edit / remove / make active).
- **Env var:** `EYE_LABELLER_SAM_HELA_LOCAL_PATH=/full/path/to/best.pt`
  before launching — adopted into the registry on first run.
- **Installer prompt:** the deploy scripts ask for the path during
  install and register it for you.

You can also just launch and annotate without any model — only the
one-click **SAM Box** is disabled until one is registered (the first
explicit SAM use will also offer a file picker).
