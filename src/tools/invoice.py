from langchain_core.tools import tool

from ._utils import encode_json, not_found, query_rows, validate_id


@tool
def get_invoices_by_customer_sorted_by_date(customer_id: str) -> str:
    """List a customer's invoices, newest first, with stable date tie-breaking."""
    numeric_id, error = validate_id(customer_id, "customer_id")
    if error:
        return error

    rows, error = query_rows(
        """
        SELECT InvoiceId AS invoice_id, InvoiceDate AS invoice_date,
               BillingAddress AS billing_address, BillingCity AS billing_city,
               BillingState AS billing_state, BillingCountry AS billing_country,
               BillingPostalCode AS billing_postal_code, Total AS total
        FROM Invoice
        WHERE CustomerId = :customer_id
        ORDER BY InvoiceDate DESC, InvoiceId DESC
        """,
        {"customer_id": numeric_id},
    )
    if error:
        return error
    if not rows:
        return not_found(f"No invoices found for customer ID: {customer_id}")
    return encode_json(rows)


@tool
def get_tracks_by_price(customer_id: str) -> str:
    """List each track purchased by a customer, most expensive first."""
    numeric_id, error = validate_id(customer_id, "customer_id")
    if error:
        return error

    rows, error = query_rows(
        """
        SELECT InvoiceLine.InvoiceLineId AS invoice_line_id,
               InvoiceLine.InvoiceId AS invoice_id,
               Track.TrackId AS track_id, Track.Name AS title,
               Artist.Name AS artist, InvoiceLine.UnitPrice AS unit_price,
               InvoiceLine.Quantity AS quantity
        FROM InvoiceLine
        JOIN Invoice ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        JOIN Track ON Track.TrackId = InvoiceLine.TrackId
        JOIN Album ON Album.AlbumId = Track.AlbumId
        JOIN Artist ON Artist.ArtistId = Album.ArtistId
        WHERE Invoice.CustomerId = :customer_id
        ORDER BY InvoiceLine.UnitPrice DESC, Track.Name, Track.TrackId,
                 InvoiceLine.InvoiceLineId
        """,
        {"customer_id": numeric_id},
    )
    if error:
        return error
    if not rows:
        return not_found(f"No purchased tracks found for customer ID: {customer_id}")
    return encode_json(rows)


@tool
def get_employee_by_invoice_and_customer(invoice_id: str, customer_id: str) -> str:
    """Find the support representative for an invoice belonging to the customer."""
    numeric_invoice_id, error = validate_id(invoice_id, "invoice_id")
    if error:
        return error
    numeric_customer_id, error = validate_id(customer_id, "customer_id")
    if error:
        return error

    rows, error = query_rows(
        """
        SELECT Employee.EmployeeId AS employee_id,
               Employee.FirstName AS first_name,
               Employee.LastName AS last_name,
               Employee.Title AS title, Employee.Email AS email,
               Invoice.InvoiceId AS invoice_id
        FROM Invoice
        JOIN Customer ON Customer.CustomerId = Invoice.CustomerId
        JOIN Employee ON Employee.EmployeeId = Customer.SupportRepId
        WHERE Invoice.InvoiceId = :invoice_id
          AND Invoice.CustomerId = :customer_id
        """,
        {"invoice_id": numeric_invoice_id, "customer_id": numeric_customer_id},
    )
    if error:
        return error
    if not rows:
        return not_found("Invoice not found for your account")
    return encode_json(rows[0])


@tool
def get_invoice_line_items(invoice_id: str, customer_id: str) -> str:
    """Get detailed line items for an invoice owned by the specified customer."""
    numeric_invoice_id, error = validate_id(invoice_id, "invoice_id")
    if error:
        return error
    numeric_customer_id, error = validate_id(customer_id, "customer_id")
    if error:
        return error

    rows, error = query_rows(
        """
        SELECT InvoiceLine.InvoiceLineId AS invoice_line_id,
               InvoiceLine.InvoiceId AS invoice_id,
               Track.TrackId AS track_id, Track.Name AS title,
               Artist.Name AS artist, Album.Title AS album,
               InvoiceLine.UnitPrice AS unit_price,
               InvoiceLine.Quantity AS quantity
        FROM InvoiceLine
        JOIN Invoice ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        JOIN Track ON Track.TrackId = InvoiceLine.TrackId
        JOIN Album ON Album.AlbumId = Track.AlbumId
        JOIN Artist ON Artist.ArtistId = Album.ArtistId
        WHERE InvoiceLine.InvoiceId = :invoice_id
          AND Invoice.CustomerId = :customer_id
        ORDER BY InvoiceLine.InvoiceLineId
        """,
        {"invoice_id": numeric_invoice_id, "customer_id": numeric_customer_id},
    )
    if error:
        return error
    if not rows:
        return not_found("Invoice not found for your account")
    return encode_json(rows)
