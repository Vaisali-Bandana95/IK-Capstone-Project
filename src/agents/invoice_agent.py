import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.config import settings
from src.tools.invoice import (
    get_employee_by_invoice_and_customer,
    get_invoice_line_items,
    get_invoices_by_customer_sorted_by_date,
    get_tracks_by_price,
)

from .prompts import INVOICE_AGENT_PROMPT


def _verified_customer_id(config: RunnableConfig) -> str | None:
    raw_id = config.get("configurable", {}).get("verified_customer_id")
    if isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id > 0:
        return str(raw_id)
    if isinstance(raw_id, str) and raw_id.isdecimal() and int(raw_id) > 0:
        return raw_id
    return None


def _missing_identity() -> str:
    return json.dumps(
        {
            "error": "Verified customer identity is unavailable",
            "code": "VERIFICATION_REQUIRED",
        }
    )


def build_invoice_tools():
    @tool("get_invoices_by_customer_sorted_by_date")
    def invoices(config: RunnableConfig) -> str:
        """List invoices for the verified customer, newest first."""
        customer_id = _verified_customer_id(config)
        if customer_id is None:
            return _missing_identity()
        return get_invoices_by_customer_sorted_by_date.invoke(
            {"customer_id": customer_id}
        )

    @tool("get_tracks_by_price")
    def purchased_tracks(config: RunnableConfig) -> str:
        """List purchased tracks for the verified customer, highest unit price first."""
        customer_id = _verified_customer_id(config)
        if customer_id is None:
            return _missing_identity()
        return get_tracks_by_price.invoke({"customer_id": customer_id})

    @tool("get_employee_by_invoice_and_customer")
    def invoice_support_rep(invoice_id: str, config: RunnableConfig) -> str:
        """Find the support representative for one of the verified customer's invoices."""
        customer_id = _verified_customer_id(config)
        if customer_id is None:
            return _missing_identity()
        return get_employee_by_invoice_and_customer.invoke(
            {"invoice_id": invoice_id, "customer_id": customer_id}
        )

    @tool("get_invoice_line_items")
    def invoice_line_items(invoice_id: str, config: RunnableConfig) -> str:
        """List line items for one of the verified customer's invoices."""
        customer_id = _verified_customer_id(config)
        if customer_id is None:
            return _missing_identity()
        return get_invoice_line_items.invoke(
            {"invoice_id": invoice_id, "customer_id": customer_id}
        )

    return [invoices, purchased_tracks, invoice_support_rep, invoice_line_items]


def _configured_model(model=None):
    if model is not None:
        return model
    if not settings.api_key:
        raise ValueError("OPENAI_API_KEY must be configured for the invoice agent")
    if not settings.model_name:
        raise ValueError("OPENAI_MODEL must be configured for the invoice agent")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=settings.api_key,
        model=settings.model_name,
        temperature=settings.temperature,
        base_url=settings.api_base_url,
    )


def build_invoice_agent(model=None):
    return create_react_agent(
        _configured_model(model),
        build_invoice_tools(),
        prompt=INVOICE_AGENT_PROMPT,
        name="invoice_agent",
    )
