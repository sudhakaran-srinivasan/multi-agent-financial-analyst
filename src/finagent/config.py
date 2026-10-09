"""Central settings for the research agent.

Agent Design: every tunable number and every secret lookup lives here, so
the rest of the code never calls ``os.environ`` or hard-codes a model name.

Why one file?
    * One place to change the model, the step cap or the token limit.
    * ``.env`` is loaded once, here, instead of in every notebook cell.
    * A missing API key fails early with a message that names the key.

AI DISCLOSURE: 
"""

import os

from dotenv import find_dotenv, load_dotenv

# --- Model settings (not secrets, so they are plain constants) -------------

# AUTHOR: confirm. Pinned so results are reproducible; "latest" aliases can
# change behaviour between runs and break the notebook demo.
MODEL: str = "openai/gpt-4o-mini"

# OpenRouter exposes an OpenAI-compatible API, so the ``openai`` SDK works
# when pointed at this URL.
BASE_URL: str = "https://openrouter.ai/api/v1"

# AUTHOR: confirm. One "step" is one model turn. A normal NVDA run needs
# about 5 tool calls + 1 final turn; the rest is room for retries. The cap
# exists to stop a looping model, not to save API calls.
MAX_STEPS: int = 8

# Upper bound on tokens the model may write per turn. Also avoids OpenRouter
# rejecting requests when the account has limited credit.
MAX_TOKENS: int = 1500

# AUTHOR: confirm. 0 makes tool choices as repeatable as the model allows,
# which keeps the notebook demo and the traces reproducible.
TEMPERATURE: float = 0.0

# Names of the secrets we expect in the environment (values live in .env).
OPENROUTER_KEY_NAME: str = "OPENROUTER_API_KEY"
FINNHUB_KEY_NAME: str = "FINNHUB_API_KEY"


def load_environment() -> None:
    """Load variables from the nearest ``.env`` file, if there is one.

    ``usecwd=True`` makes the search start from the current working folder.
    Without it, ``find_dotenv`` starts from the calling file's folder, which
    is why a notebook could not find the repo's ``.env``.

    Safe to call many times. Existing environment variables win over the
    file, so a real shell export is never silently overwritten.
    """
    load_dotenv(find_dotenv(usecwd=True))


def require_env(name: str) -> str:
    """Return the environment variable ``name`` or fail with a clear message.

    Args:
        name: The variable to read, e.g. ``"FINNHUB_API_KEY"``.

    Raises:
        RuntimeError: If it is missing or blank. The message names the
            variable so the fix is obvious.
    """
    load_environment()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to your .env file "
            f"(see .env.example) or export it in your shell."
        )
    return value