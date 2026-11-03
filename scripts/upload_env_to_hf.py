#!/usr/bin/env python3
"""
Upload a PolaRiS environment folder to the Hugging Face dataset repository
`PolaRiS-Evals/PolaRiS-Hub` after performing local validation.
Validation tries to catch the most common mistakes (missing assets, malformed
`initial_conditions.json`, unreadable USD), but it cannot guarantee runtime
success inside Isaac Sim. Use this as a fast client-side gate before pushing.
# Example commands:
#   uv run scripts/upload_env_to_hf.py ./PolaRiS-Hub/food_bussing --dry-run
"""

from polaris.hf_upload import main  # type: ignore


if __name__ == "__main__":
    main()
