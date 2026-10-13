import json

import pytest
from pydantic import BaseModel

from llama_course.shop_tools import (
    SHOP_TOOLS,
    ConvertArgs,
    ShippingArgs,
    StockArgs,
    convert_units,
    get_stock,
    shipping_cost,
)
from llama_course.tools import (
    NO_TOOL,
    Tool,
    ToolBox,
    ToolCallFailed,
    ToolError,
    ask,
    ask_multi,
)


def call(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def test_valid_call_runs_the_tool_and_answers_from_its_result(fake_llm):
    llm = fake_llm([call("get_stock", item="helmet"), "We have 40 helmets."])
    result = ask(llm, SHOP_TOOLS, "How many helmets?")
    assert (result.tool, result.arguments, result.text) == ("get_stock", {"item": "helmet"}, "We have 40 helmets.")
    assert result.result == "40 helmet(s) in stock"
    final = llm.calls[1]["messages"]
    assert "Tool result: 40 helmet(s) in stock" in final[-1]["content"]
    assert "JSON" not in final[0]["content"]  # fresh chat, not the JSON-only tool prompt
    assert "response_format" in llm.calls[0] and "response_format" not in llm.calls[1]


def test_no_tool_answers_directly(fake_llm):
    llm = fake_llm([json.dumps({"tool": NO_TOOL}), "Paris."])
    result = ask(llm, SHOP_TOOLS, "Capital of France?")
    assert result.tool is None and result.text == "Paris." and len(llm.calls) == 2


def test_unknown_tool_is_reported_to_the_model_and_retried(fake_llm):
    llm = fake_llm([call("delete_everything"), call("get_stock", item="lock"), "25 locks."])
    result = ask(llm, SHOP_TOOLS, "Locks?")
    assert result.tool == "get_stock"
    feedback = llm.calls[1]["messages"][-1]["content"]
    assert "Unknown tool 'delete_everything'" in feedback and "get_stock" in feedback


def test_bad_arguments_are_reported_and_retried(fake_llm):
    llm = fake_llm([call("shipping_cost", order_total="lots"), call("shipping_cost", order_total=900), "$29."])
    result = ask(llm, SHOP_TOOLS, "Shipping for 900?")
    assert result.result == "shipping costs $29.00"
    assert "order_total" in llm.calls[1]["messages"][-1]["content"]


def test_malformed_json_is_retried(fake_llm):
    llm = fake_llm(["{not json", call("get_stock", item="lock"), "ok"])
    assert ask(llm, SHOP_TOOLS, "Locks?").tool == "get_stock"


def test_gives_up_after_the_retry_limit(fake_llm):
    llm = fake_llm([call("nope"), call("nope")])
    with pytest.raises(ToolCallFailed, match="2 attempts"):
        ask(llm, SHOP_TOOLS, "x", retries=1)
    assert len(llm.calls) == 2


def test_tool_error_becomes_a_result_the_model_can_explain(fake_llm):
    llm = fake_llm([call("get_stock", item="unicycle"), "We don't sell that."])
    result = ask(llm, SHOP_TOOLS, "Unicycles?")
    assert result.result.startswith("Error: unknown item 'unicycle'")


def test_schema_offers_each_tool_and_no_tool():
    options = SHOP_TOOLS.schema()["anyOf"]
    names = [o["properties"]["tool"]["const"] for o in options]
    assert names == ["get_stock", "convert_units", "shipping_cost", NO_TOOL]
    assert options[2]["properties"]["arguments"]["properties"]["order_total"]["minimum"] == 0


def test_system_prompt_lists_tools():
    prompt = SHOP_TOOLS.system_prompt()
    assert "get_stock" in prompt and "convert_units" in prompt and NO_TOOL in prompt


def test_negative_order_total_is_rejected_before_running():
    with pytest.raises(Exception, match="order_total"):
        SHOP_TOOLS.call("shipping_cost", {"order_total": -5})


def test_a_tool_only_runs_if_registered():
    ran = []

    class Args(BaseModel):
        x: int

    box = ToolBox([Tool("safe", "d", Args, lambda a: ran.append(a.x) or "ok")])
    with pytest.raises(KeyError):
        box.call("other", {"x": 1})
    assert ran == []


def test_shop_tools():
    assert get_stock(StockArgs(item=" Helmet ")) == "40 helmet(s) in stock"
    assert get_stock(StockArgs(item="helmets")) == "40 helmet(s) in stock"
    assert convert_units(ConvertArgs(value=10, from_unit="km", to_unit="miles")) == "6.21 miles"
    assert convert_units(ConvertArgs(value=2, from_unit="kg", to_unit="lb")) == "4.41 lb"
    with pytest.raises(ToolError):
        convert_units(ConvertArgs(value=1, from_unit="km", to_unit="kg"))
    assert shipping_cost(ShippingArgs(order_total=1600)) == "shipping costs $0.00"
    assert shipping_cost(ShippingArgs(order_total=1600, express=True)) == "shipping costs $79.00"
    assert shipping_cost(ShippingArgs(order_total=100)) == "shipping costs $29.00"


def test_ask_multi_chains_calls_and_answers_from_every_result(fake_llm):
    llm = fake_llm([call("get_stock", item="helmet"), call("get_stock", item="lock"), json.dumps({"tool": NO_TOOL}), "Both."])
    result = ask_multi(llm, SHOP_TOOLS, "Helmets and locks?")
    assert [s[2] for s in result.steps] == ["40 helmet(s) in stock", "25 lock(s) in stock"]
    assert (result.tool, result.arguments, result.text) == ("get_stock", {"item": "lock"}, "Both.")
    final = llm.calls[-1]["messages"][-1]["content"]
    assert "40 helmet(s)" in final and "25 lock(s)" in final


def test_ask_multi_reports_a_bad_call_and_continues(fake_llm):
    llm = fake_llm([call("delete_everything"), call("get_stock", item="lock"), json.dumps({"tool": NO_TOOL}), "25."])
    result = ask_multi(llm, SHOP_TOOLS, "Locks?")
    assert result.tool == "get_stock" and len(result.steps) == 1
    assert "Unknown tool 'delete_everything'" in llm.calls[1]["messages"][-1]["content"]


def test_ask_multi_stops_at_max_turns(fake_llm):
    llm = fake_llm([call("get_stock", item="lock")] * 2 + ["25."])
    result = ask_multi(llm, SHOP_TOOLS, "Locks?", max_turns=2)
    assert len(result.steps) == 2 and len(llm.calls) == 3


def test_ask_multi_gives_up_when_no_call_is_valid(fake_llm):
    llm = fake_llm(["{bad", call("nope")])
    with pytest.raises(ToolCallFailed, match="2 turns"):
        ask_multi(llm, SHOP_TOOLS, "x", max_turns=2)
