"""Offline tests for the tool menu (finagent.tools.tool_specs).

Definition of "this works": the menu the model reads and the functions the
dispatcher runs can never drift apart, every entry is well-formed, and the
profile tool stays off the menu (the loop runs it first).
"""

from finagent.tools.news import get_news_insights
from finagent.tools.tool_specs import TOOL_FUNCTIONS, TOOL_SPECS


def _names():
    return [spec["function"]["name"] for spec in TOOL_SPECS]


def test_menu_and_function_map_have_the_same_names():
    assert sorted(_names()) == sorted(TOOL_FUNCTIONS)


def test_names_are_unique():
    assert len(_names()) == len(set(_names()))


def test_every_entry_is_well_formed():
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        function = spec["function"]
        assert function["description"].strip()
        params = function["parameters"]
        assert params["type"] == "object"
        assert params["required"] == ["ticker"]
        assert params["additionalProperties"] is False
        assert params["properties"]["ticker"]["type"] == "string"


def test_profile_is_not_on_the_menu():
    assert "get_company_profile" not in _names()


def test_news_stub_returns_an_empty_list_not_mock_text():
    assert get_news_insights("NVDA") == []