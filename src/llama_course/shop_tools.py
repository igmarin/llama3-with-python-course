"""Three safe example tools for the tool-calling lesson (a made-up bike shop).

All are pure functions over in-memory data: no files, no network, no shell.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from llama_course.tools import Tool, ToolBox, ToolError

STOCK = {"city bike": 12, "mountain bike": 5, "road bike": 0, "helmet": 40, "lock": 25}

Unit = Literal["km", "miles", "kg", "lb"]
_FACTORS = {("km", "miles"): 0.621371, ("miles", "km"): 1.609344, ("kg", "lb"): 2.204623, ("lb", "kg"): 0.453592}


class StockArgs(BaseModel):
    item: str = Field(description="Product name, for example 'helmet'")


class ConvertArgs(BaseModel):
    value: float
    from_unit: Unit
    to_unit: Unit


class ShippingArgs(BaseModel):
    order_total: float = Field(ge=0, description="Order total in dollars")
    express: bool = False


def get_stock(args: StockArgs) -> str:
    """Return how many units of an item are in stock."""
    item = args.item.strip().lower()
    if item not in STOCK and item.removesuffix("s") in STOCK:  # "helmets" -> "helmet"
        item = item.removesuffix("s")
    if item not in STOCK:
        raise ToolError(f"unknown item {args.item!r}. Known items: {', '.join(STOCK)}")
    return f"{STOCK[item]} {item}(s) in stock"


def convert_units(args: ConvertArgs) -> str:
    """Convert km/miles or kg/lb."""
    if args.from_unit == args.to_unit:
        return f"{args.value:g} {args.to_unit}"
    factor = _FACTORS.get((args.from_unit, args.to_unit))
    if factor is None:
        raise ToolError(f"cannot convert {args.from_unit} to {args.to_unit}")
    return f"{args.value * factor:.2f} {args.to_unit}"


def shipping_cost(args: ShippingArgs) -> str:
    """Express is $79. Standard is $29, and free for orders over $1,500."""
    if args.express:
        cost = 79
    else:
        cost = 0 if args.order_total > 1500 else 29
    return f"shipping costs ${cost:.2f}"


SHOP_TOOLS = ToolBox(
    [
        Tool("get_stock", "Look up how many units of a product are in stock.", StockArgs, get_stock),
        Tool("convert_units", "Convert between km and miles, or kg and lb.", ConvertArgs, convert_units),
        Tool("shipping_cost", "Work out the shipping cost for an order.", ShippingArgs, shipping_cost),
    ]
)
