"""The tool menu the model reads, and the map from tool name to function.

Agent Design: the model never sees our Python. It sees only the text below
(a name, a description and the parameters) and decides from that which tool
to call. So the descriptions are prompts, not comments: they say what comes
back and when to use the tool.

Company profile is deliberately NOT on the menu. The loop runs it first, in
Python, as a cheap check that the ticker is a real stock. Its result is
handed to the model as starting context.

Format: the OpenAI "function calling" schema, which OpenRouter accepts.

AI DISCLOSURE: 
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from finagent.tools.earnings_history import get_earnings_history
from finagent.tools.fundamentals import get_fundamentals
from finagent.tools.news import get_news_insights

_TICKER_PARAMETER: dict[str, Any] = {
    "type": "string",
    "description": "Stock ticker symbol, for example NVDA.",
}


def _spec(name: str, description: str) -> dict[str, Any]:
    """Build one menu entry. Every tool takes a single ``ticker`` argument.

    ``additionalProperties: False`` tells the model not to invent extra
    arguments; the dispatcher also rejects them, so a model that ignores
    the hint still cannot sneak anything through.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"ticker": _TICKER_PARAMETER},
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    }


# AUTHOR: review these descriptions. They decide when the model picks a tool.
TOOL_SPECS: list[dict[str, Any]] = [
    _spec(
        "get_fundamentals",
        "Key financial ratios for a stock, grouped as valuation (such as "
        "trailing and forward P/E), profitability margins, growth, balance "
        "sheet (debt and cash), price levels and analyst sentiment. A null "
        "value means the data was missing or invalid; the reason is listed "
        "under 'gaps'. Use it for any question about whether a stock is "
        "cheap, profitable or financially healthy.",
    ),
    _spec(
        "get_earnings_history",
        "The last up to 4 quarters of reported earnings per share versus "
        "analyst estimates. Each quarter is classified beat, in-line or "
        "miss, with the surprise percent, plus a summary (beats, misses, "
        "beat rate, average surprise). Use it to judge whether the company "
        "usually meets or beats expectations.",
    ),
    _spec(
        "get_news_insights",
        "Recent news about the company, each item classified as earnings, "
        "general or market news, with a short summary and a positive, "
        "neutral or negative sentiment. An empty list means no news "
        "insights are available, which should be reported as a data gap.",
    ),
]

# Name -> real function. The dispatcher uses this to run what the model asks
# for. A test checks that its keys always match the menu above.
TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_fundamentals": get_fundamentals,
    "get_earnings_history": get_earnings_history,
    "get_news_insights": get_news_insights,
}