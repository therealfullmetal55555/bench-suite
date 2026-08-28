"""The invoices the extraction family is built from, and the renderer that turns each one
into a document an agent has to read.

Structured data with a renderer rather than a directory of sixteen text files. Two reasons,
and the second is the one that matters:

* Sixteen more files in the repository is sixteen more things to keep in sync with the
  expectations, and an extraction benchmark where the document and the expected value can
  drift apart is a benchmark that fails everybody the first time somebody tidies a comma.
* Here the document *is* generated from the values, so the two cannot disagree. A test
  asserts exactly that — every field value appears in the rendered text — and it is the
  reason `render` is a pure function rather than a fixture.

The noise is deliberate. Real invoices say "Net 30", put the total under a heading, spell
the date out in one document and shorten it in the next. An extractor that works on
`Invoice 4471 | total 77.50` and fails on `Total due: €77,50` has learned the fixture, not
the task, so the same values are dressed eight different ways here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CURRENCIES = ("EUR", "USD", "GBP", "SEK")

#: Field order for the rendered table, and the label each field gets when it has one.
LABELS = {
    "invoice_id": "Invoice",
    "vendor": "From",
    "issued": "Issued",
    "due": "Due",
    "net": "Subtotal",
    "tax": "VAT",
    "total": "Total",
    "po_number": "PO",
}


@dataclass(frozen=True)
class Invoice:
    """One document: the values, and how they are dressed."""

    key: str
    fields: dict[str, Any]
    """The ground truth. Numbers are numbers, dates are ISO strings, and `total` is what the
    document says is owed — which is not always `net + tax`, because a credit note is a
    document too."""

    style: dict[str, Any] = field(default_factory=dict)
    """`symbol`, `date_format`, `greeting`, `note`, `shouty`… The knobs that make two
    documents with the same fields look like they came from two companies."""

    def values(self) -> dict[str, Any]:
        return dict(self.fields)

    def schema_fields(self) -> list[str]:
        """Which fields a schema expectation should require: the ones a person would say
        are non-negotiable for the document, which is not the same as all of them."""
        required = ["vendor", "total", "currency"]
        if "invoice_id" in self.fields:
            required.insert(0, "invoice_id")
        return required


def render(invoice: Invoice) -> str:
    """The document text, as a person would receive it.

    Formatted through the per-invoice style rather than through one template, because the
    point of the family is that the *shape* of a document is not fixed. `format_value` is
    the single place that knows how a number gets spelled, and it is used for both the
    document and the tests' consistency check.
    """
    style = invoice.style
    symbol = style.get("symbol", "")
    date_format = style.get("date_format", "iso")
    lines: list[str] = []

    greeting = style.get("greeting")
    if greeting:
        lines.append(greeting)
        lines.append("")

    if style.get("shouty"):
        lines.append(f"INVOICE {invoice.fields.get('invoice_id', '')}".strip())
    else:
        lines.append(f"{LABELS['invoice_id']} {invoice.fields.get('invoice_id', '')}".strip())
    lines.append(f"{LABELS['vendor']}: {invoice.fields['vendor']}")
    if "issued" in invoice.fields:
        lines.append(f"{LABELS['issued']}: {format_value(invoice.fields['issued'], date_format)}")
    if "due" in invoice.fields:
        lines.append(f"{LABELS['due']}: {format_value(invoice.fields['due'], date_format)}")
    if "po_number" in invoice.fields:
        lines.append(f"{LABELS['po_number']}: {invoice.fields['po_number']}")
    lines.append("")

    for line in style.get("items", []):
        lines.append(f"  {line}")
    if style.get("items"):
        lines.append("")

    if "net" in invoice.fields:
        lines.append(f"{LABELS['net']}: {format_value(invoice.fields['net'], 'money', symbol)}")
    if "tax" in invoice.fields:
        lines.append(f"{LABELS['tax']}: {format_value(invoice.fields['tax'], 'money', symbol)}")
    lines.append(f"{LABELS['total']}: {format_value(invoice.fields['total'], 'money', symbol)}")
    if invoice.fields.get("credit"):
        lines.append("  this is a credit note: the amount is owed back to the customer")
    lines.append("")

    note = style.get("note")
    if note:
        lines.append(note)
    terms = style.get("terms")
    if terms:
        lines.append(terms)
    return "\n".join(lines).rstrip() + "\n"


def format_value(value: Any, style: str, symbol: str = "") -> str:
    """How a value appears in a document.

    Three date spellings and two number spellings, because that is what turns up: a comma
    decimal separator is standard in half of Europe, and an extractor that returns
    `"77,50"` as a string has failed a numeric field in a way a downstream system notices
    only at the invoicing step.
    """
    if style == "money":
        number = float(value)
        text = f"{number:,.2f}"
        return f"{symbol}{text}" if symbol else text
    if style == "iso":
        return str(value)
    if style == "long":
        year, month, day = str(value).split("-")
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        return f"{int(day)} {months[int(month) - 1]} {year}"
    if style == "dotted":
        year, month, day = str(value).split("-")
        return f"{day}.{month}.{year}"
    return str(value)


def documents() -> list[Invoice]:
    """The sixteen documents, hand-written and deliberately uneven.

    The first eight are clean, single-currency, one-page invoices: they exist so that a
    naive implementation gets a score above zero and a reader can see partial credit doing
    its job. The next eight are the ones that separate extractors — a credit note whose
    total is negative, a document with no invoice number at all, a page where the VAT is
    stated in the prose instead of a line, and one where the subtotal and the total disagree
    because a discount line sits between them.
    """
    return [
        Invoice(
            "inv-0001",
            {
                "invoice_id": "4471",
                "vendor": "Northwind Tools",
                "issued": "2026-03-04",
                "due": "2026-04-03",
                "net": 64.58,
                "tax": 12.92,
                "total": 77.50,
                "currency": "EUR",
            },
            {
                "symbol": "€",
                "date_format": "long",
                "items": ["2 × clamp, 24.00", "1 × socket set, 16.58"],
            },
        ),
        Invoice(
            "inv-0002",
            {
                "invoice_id": "88213",
                "vendor": "Helio Freight",
                "issued": "2026-03-11",
                "net": 240.00,
                "tax": 48.00,
                "total": 288.00,
                "currency": "USD",
            },
            {
                "symbol": "$",
                "date_format": "iso",
                "terms": "Net 30.",
                "items": ["4 × pallet, 60.00"],
            },
        ),
        Invoice(
            "inv-0003",
            {
                "invoice_id": "A-2291",
                "vendor": "Norrsken Tryck",
                "issued": "2026-02-19",
                "due": "2026-03-19",
                "net": 1750.00,
                "tax": 437.50,
                "total": 2187.50,
                "currency": "SEK",
            },
            {"date_format": "dotted", "items": ["print run, 1400.00", "paper surcharge, 350.00"]},
        ),
        Invoice(
            "inv-0004",
            {
                "invoice_id": "77102-B",
                "vendor": "Copperfield Ltd",
                "issued": "2026-01-30",
                "net": 99.99,
                "tax": 0.00,
                "total": 99.99,
                "currency": "GBP",
                "po_number": "PO-4417",
            },
            {"symbol": "£", "date_format": "long", "terms": "Zero-rated: reverse charge applies."},
        ),
        Invoice(
            "inv-0005",
            {
                "invoice_id": "INV-2026-0033",
                "vendor": "Bridge Analytics",
                "issued": "2026-04-02",
                "due": "2026-05-02",
                "net": 1200.00,
                "tax": 300.00,
                "total": 1500.00,
                "currency": "EUR",
            },
            {
                "greeting": "Dear accounts team,",
                "items": ["dashboard licence, 900.00", "onboarding, 300.00"],
                "terms": "Payment due within 30 days.",
            },
        ),
        Invoice(
            "inv-0006",
            {
                "invoice_id": "55014",
                "vendor": "Kestrel Couriers",
                "issued": "2026-04-18",
                "net": 42.00,
                "tax": 8.40,
                "total": 50.40,
                "currency": "EUR",
            },
            {
                "symbol": "€",
                "date_format": "dotted",
                "shouty": True,
                "items": ["same-day delivery, 42.00"],
            },
        ),
        Invoice(
            "inv-0007",
            {
                "invoice_id": "X-9",
                "vendor": "Lumen Electrical",
                "issued": "2026-05-06",
                "due": "2026-06-05",
                "net": 310.00,
                "tax": 77.50,
                "total": 387.50,
                "currency": "USD",
                "po_number": "4417-A",
            },
            {
                "symbol": "$",
                "date_format": "long",
                "note": "Questions about this invoice: billing@lumen.example.",
            },
        ),
        Invoice(
            "inv-0008",
            {
                "invoice_id": "0314",
                "vendor": "Marlow & Sons",
                "issued": "2026-05-21",
                "net": 5.50,
                "tax": 1.10,
                "total": 6.60,
                "currency": "GBP",
            },
            {"symbol": "£", "date_format": "iso", "items": ["1 × label roll, 5.50"]},
        ),
        # --- from here on, the documents that separate extractors -------------
        Invoice(
            "inv-0009",
            {
                "invoice_id": "CN-1102",
                "vendor": "Northwind Tools",
                "issued": "2026-03-20",
                "net": -64.58,
                "tax": -12.92,
                "total": -77.50,
                "currency": "EUR",
                "credit": True,
            },
            {"symbol": "€", "date_format": "long", "note": "Credit for invoice 4471."},
        ),
        Invoice(
            "inv-0010",
            {
                "vendor": "Bramblewood Supplies",
                "issued": "2026-03-27",
                "net": 220.00,
                "tax": 55.00,
                "total": 275.00,
                "currency": "EUR",
            },
            {
                "symbol": "€",
                "date_format": "long",
                "note": "No invoice number has been assigned to this document.",
                "items": ["replacement parts, 220.00"],
            },
        ),
        Invoice(
            "inv-0011",
            {
                "invoice_id": "70041",
                "vendor": "Fenwick Print",
                "issued": "2026-04-09",
                "net": 480.00,
                "tax": 115.20,
                "total": 595.20,
                "currency": "EUR",
            },
            {
                "symbol": "€",
                "date_format": "dotted",
                # The VAT is stated in prose and not as a line. An extractor that reads the
                # first number next to "VAT" still gets it right; one that only reads a
                # table does not.
                "note": "VAT at 24% is charged on the subtotal, 115.20, and is included in the "
                "total.",
            },
        ),
        Invoice(
            "inv-0012",
            {
                "invoice_id": "93005",
                "vendor": "Atlas Hardware",
                "issued": "2026-04-23",
                "due": "2026-05-23",
                "net": 400.00,
                "tax": 96.00,
                "total": 472.00,
                "currency": "EUR",
                "discount": 24.00,
            },
            {
                "symbol": "€",
                "date_format": "long",
                "items": ["parts, 400.00", "loyalty discount, -24.00"],
                "note": "The total reflects a 24.00 discount taken off the subtotal.",
            },
        ),
        Invoice(
            "inv-0013",
            {
                "invoice_id": "INV/2026/88",
                "vendor": "Meridian Cloud",
                "issued": "2026-05-01",
                "due": "2026-05-31",
                "net": 3250.00,
                "tax": 812.50,
                "total": 4062.50,
                "currency": "USD",
            },
            {
                "greeting": "Statement for the period 1–31 May 2026.",
                "items": ["compute, 2100.00", "storage, 700.00", "egress, 450.00"],
                "terms": "Payable within 30 days of issue.",
            },
        ),
        Invoice(
            "inv-0014",
            {
                "invoice_id": "5541",
                "vendor": "Sable Catering",
                "issued": "2026-05-14",
                "net": 810.00,
                "tax": 202.50,
                "total": 1012.50,
                "currency": "SEK",
                "po_number": "PO-8801",
            },
            {"date_format": "dotted", "items": ["lunch service, 810.00"]},
        ),
        Invoice(
            "inv-0015",
            {
                "invoice_id": "b-2026-0777",
                "vendor": "Tidewater Legal",
                "issued": "2026-06-02",
                "due": "2026-07-02",
                "net": 5000.00,
                "tax": 1250.00,
                "total": 6250.00,
                "currency": "GBP",
                "po_number": "4417-B",
            },
            {
                "symbol": "£",
                "date_format": "long",
                "shouty": True,
                "terms": "Late payment interest may apply.",
            },
        ),
        Invoice(
            "inv-0016",
            {
                "invoice_id": "2200-16",
                "vendor": "Ridgeway Coffee",
                "issued": "2026-06-11",
                "net": 96.00,
                "tax": 23.04,
                "total": 119.04,
                "currency": "EUR",
            },
            {
                "symbol": "€",
                "date_format": "iso",
                "items": ["office coffee, 96.00"],
                "terms": "Net 14.",
            },
        ),
    ]
