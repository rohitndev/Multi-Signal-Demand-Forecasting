"""
agent.py
--------
Replenishment agent: looks at a P50 forecast plus current inventory and decides
whether to place a purchase order. Uses LangChain + Groq (llama3-8b-8192) to
draft a short, human-readable recommendation message.

The numeric decision is deterministic Python (so it always works offline); only
the wording of the message comes from the LLM, with a plain-text fallback.

Run:
    python -m src.agent
"""
from __future__ import annotations

from src.config import get_env


def _decide(p50_forecast: list[float], current_inventory: int, reorder_point: int) -> dict:
    """
    Deterministic ordering logic.

    Rule (from the spec):
      * If total forecast demand > 80% of current inventory -> urgent order
      * Else if inventory has fallen to/below the reorder point -> standard order
      * Else -> no action
    Recommended qty covers projected demand minus what's on hand (never negative).
    """
    total_demand = float(sum(p50_forecast))
    recommended_qty = max(0, int(round(total_demand - current_inventory)))

    if total_demand > current_inventory * 0.8:
        action = "urgent"
        # Top up to cover demand plus a 20% safety buffer.
        recommended_qty = max(recommended_qty, int(round(total_demand * 1.2 - current_inventory)))
    elif current_inventory <= reorder_point:
        action = "standard"
    else:
        action = "none"
        recommended_qty = 0

    return {
        "action": action,
        "recommended_qty": recommended_qty,
        "total_demand": round(total_demand, 2),
    }


def _draft_message(store_id: int, decision: dict, current_inventory: int) -> str:
    """Ask Groq for a 3-line PO recommendation; fall back to a template offline."""
    fallback = (
        f"Store {store_id} replenishment: {decision['action'].upper()}.\n"
        f"Projected 28-day demand is {decision['total_demand']} units vs "
        f"{current_inventory} on hand.\n"
        f"Recommended purchase order quantity: {decision['recommended_qty']} units."
    )

    api_key = get_env("GROQ_API_KEY")
    if not api_key:
        print("[agent] No GROQ_API_KEY set -> using template message.")
        return fallback

    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, api_key=api_key)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a retail supply-chain assistant. Write a concise, "
                    "professional 3-line purchase order recommendation. No preamble.",
                ),
                (
                    "human",
                    "Store {store_id}. Action: {action}. "
                    "Projected 28-day demand: {demand} units. "
                    "Current inventory: {inventory} units. "
                    "Recommended order quantity: {qty} units. "
                    "Write exactly 3 lines.",
                ),
            ]
        )
        chain = prompt | llm
        resp = chain.invoke(
            {
                "store_id": store_id,
                "action": decision["action"],
                "demand": decision["total_demand"],
                "inventory": current_inventory,
                "qty": decision["recommended_qty"],
            }
        )
        return resp.content.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[agent] Groq call failed ({exc}) -> using template message.")
        return fallback


def replenish(
    store_id: int,
    p50_forecast: list[float],
    current_inventory: int,
    reorder_point: int = 100,
) -> dict:
    """Run the full agent: decide + draft message. Returns the recommendation dict."""
    decision = _decide(p50_forecast, current_inventory, reorder_point)
    message = _draft_message(store_id, decision, current_inventory)
    result = {
        "store_id": store_id,
        "action": decision["action"],
        "recommended_qty": decision["recommended_qty"],
        "message": message,
    }
    print("\n--- Replenishment Recommendation ---")
    print(f"Action          : {result['action']}")
    print(f"Recommended qty : {result['recommended_qty']}")
    print(f"Message         :\n{result['message']}")
    return result


if __name__ == "__main__":
    # Demo with a flat 28-day forecast of ~500 units/day.
    demo_forecast = [500.0] * 28
    replenish(store_id=1, p50_forecast=demo_forecast, current_inventory=8000, reorder_point=2000)
