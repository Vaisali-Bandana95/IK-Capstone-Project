import json

import pytest

from src.tools import (
    check_for_songs,
    get_albums_by_artist,
    get_employee_by_invoice_and_customer,
    get_invoice_line_items,
    get_invoices_by_customer_sorted_by_date,
    get_songs_by_genre,
    get_track_details,
    get_tracks_by_artist,
    get_tracks_by_price,
)


def invoke(tool, **arguments) -> object:
    result = tool.invoke(arguments)
    assert isinstance(result, str)
    return json.loads(result)


def test_all_nine_functions_are_langchain_tools() -> None:
    tools = (
        get_albums_by_artist,
        get_tracks_by_artist,
        get_songs_by_genre,
        check_for_songs,
        get_track_details,
        get_invoices_by_customer_sorted_by_date,
        get_tracks_by_price,
        get_employee_by_invoice_and_customer,
        get_invoice_line_items,
    )

    assert len(tools) == 9
    assert all(hasattr(tool, "invoke") for tool in tools)


def test_get_albums_by_artist_found() -> None:
    results = invoke(get_albums_by_artist, artist_name="AC/DC")

    assert results
    assert any(album["artist"] == "AC/DC" for album in results)


def test_get_albums_by_artist_not_found() -> None:
    result = invoke(get_albums_by_artist, artist_name="No Such Artist")

    assert "No albums found" in result["message"]


def test_get_tracks_by_artist_returns_count_and_at_most_twenty() -> None:
    result = invoke(get_tracks_by_artist, artist_name="Iron Maiden")

    assert len(result["tracks"]) <= 20
    assert result["total_count"] >= len(result["tracks"])
    assert result["truncated"] is (result["total_count"] > len(result["tracks"]))


def test_get_tracks_by_artist_not_found() -> None:
    result = invoke(get_tracks_by_artist, artist_name="No Such Artist")

    assert "No tracks found" in result["message"]


def test_get_songs_by_genre_returns_unique_artists_and_at_most_ten() -> None:
    result = invoke(get_songs_by_genre, genre="Rock")
    artists = [track["artist"] for track in result["tracks"]]

    assert 0 < len(artists) <= 10
    assert len(artists) == len(set(artists))
    assert result["total_count"] >= len(artists)
    assert result["truncated"] is (result["total_count"] > len(artists))


def test_get_songs_by_genre_is_deterministic() -> None:
    first = invoke(get_songs_by_genre, genre="Rock")
    second = invoke(get_songs_by_genre, genre="Rock")

    assert first == second


def test_get_songs_by_genre_not_found() -> None:
    result = invoke(get_songs_by_genre, genre="No Such Genre")

    assert "No songs found" in result["message"]


def test_check_for_songs_found() -> None:
    result = invoke(check_for_songs, title="for those about to rock")

    assert result["tracks"]
    assert result["total_count"] >= len(result["tracks"])


def test_check_for_songs_not_found() -> None:
    result = invoke(check_for_songs, title="No Such Song")

    assert "No songs found" in result["message"]


def test_get_track_details_found() -> None:
    result = invoke(get_track_details, track_id="1")

    assert result["track_id"] == 1
    assert result["title"]
    assert "unit_price" in result
    assert "media_type" in result


@pytest.mark.parametrize("track_id", ["abc", "1.5", "-1", "0", "2147483648"])
def test_get_track_details_rejects_invalid_ids(track_id: str) -> None:
    result = invoke(get_track_details, track_id=track_id)

    assert result["code"] == "INVALID_INPUT"
    assert "track_id" in result["error"]


def test_get_track_details_not_found() -> None:
    result = invoke(get_track_details, track_id="2147483647")

    assert "No track found" in result["message"]


def test_get_invoices_by_customer_found_and_sorted_newest_first() -> None:
    invoices = invoke(get_invoices_by_customer_sorted_by_date, customer_id="1")
    dates = [invoice["invoice_date"] for invoice in invoices]

    assert invoices
    assert dates == sorted(dates, reverse=True)


def test_get_invoices_by_customer_not_found() -> None:
    result = invoke(
        get_invoices_by_customer_sorted_by_date,
        customer_id="2147483647",
    )

    assert "No invoices found" in result["message"]


@pytest.mark.parametrize(
    "customer_id",
    ["customer", "1.2", "-1", "0", "2147483648"],
)
def test_invoice_list_rejects_invalid_customer_ids(customer_id: str) -> None:
    result = invoke(
        get_invoices_by_customer_sorted_by_date,
        customer_id=customer_id,
    )

    assert result["code"] == "INVALID_INPUT"


def test_get_tracks_by_price_returns_each_purchase_sorted_descending() -> None:
    tracks = invoke(get_tracks_by_price, customer_id="1")
    prices = [track["unit_price"] for track in tracks]

    assert tracks
    assert prices == sorted(prices, reverse=True)
    assert all("invoice_line_id" in track for track in tracks)


def test_get_tracks_by_price_not_found() -> None:
    result = invoke(get_tracks_by_price, customer_id="2147483647")

    assert "No purchased tracks found" in result["message"]


def test_get_tracks_by_price_rejects_invalid_customer_id() -> None:
    result = invoke(get_tracks_by_price, customer_id="not-a-customer")

    assert result["code"] == "INVALID_INPUT"


def test_get_employee_by_invoice_and_customer_found() -> None:
    result = invoke(
        get_employee_by_invoice_and_customer,
        invoice_id="1",
        customer_id="2",
    )

    assert result["employee_id"]
    assert result["first_name"]
    assert result["invoice_id"] == 1


def test_get_employee_by_invoice_and_customer_hides_foreign_invoice() -> None:
    result = invoke(
        get_employee_by_invoice_and_customer,
        invoice_id="1",
        customer_id="1",
    )

    assert result["message"] == "Invoice not found for your account"


@pytest.mark.parametrize(
    ("invoice_id", "customer_id"),
    [("abc", "1"), ("1", "customer"), ("0", "1"), ("1", "-1")],
)
def test_get_employee_by_invoice_rejects_invalid_ids(
    invoice_id: str,
    customer_id: str,
) -> None:
    result = invoke(
        get_employee_by_invoice_and_customer,
        invoice_id=invoice_id,
        customer_id=customer_id,
    )

    assert result["code"] == "INVALID_INPUT"


def test_get_invoice_line_items_found() -> None:
    result = invoke(get_invoice_line_items, invoice_id="1", customer_id="2")

    assert result
    assert all(item["invoice_id"] == 1 for item in result)


def test_get_invoice_line_items_hides_foreign_invoice() -> None:
    result = invoke(get_invoice_line_items, invoice_id="1", customer_id="1")

    assert result["message"] == "Invoice not found for your account"


@pytest.mark.parametrize(
    ("invoice_id", "customer_id"),
    [("abc", "2"), ("1", "customer"), ("-1", "2"), ("1", "0")],
)
def test_get_invoice_line_items_rejects_invalid_ids(
    invoice_id: str,
    customer_id: str,
) -> None:
    result = invoke(
        get_invoice_line_items,
        invoice_id=invoice_id,
        customer_id=customer_id,
    )

    assert result["code"] == "INVALID_INPUT"
