from langchain_core.tools import tool

from ._utils import (
    encode_json,
    escaped_like,
    fuzzy_search_parameter,
    log_tool_call,
    not_found,
    query_rows,
    validate_id,
)


@tool
def get_albums_by_artist(artist_name: str) -> str:
    """Find albums by a fuzzy artist-name match."""
    log_tool_call("get_albums_by_artist")
    rows, error = query_rows(
        """
        SELECT Artist.Name AS artist, Album.Title AS album, Album.AlbumId AS album_id
        FROM Artist
        JOIN Album ON Album.ArtistId = Artist.ArtistId
        WHERE LOWER(Artist.Name) LIKE LOWER(:artist) ESCAPE '\\'
        ORDER BY Artist.Name, Album.Title, Album.AlbumId
        """,
        {"artist": escaped_like(artist_name)},
    )
    if error:
        return error
    if not rows:
        suggestion, fuzzy_error = fuzzy_search_parameter(
            artist_name,
            sql="SELECT Name AS name FROM Artist",
            name="name",
        )
        if fuzzy_error:
            return fuzzy_error
        if suggestion and suggestion.casefold() != artist_name.casefold():
            return get_albums_by_artist.func(suggestion)
        return not_found(f"No albums found for artist: {artist_name}")
    return encode_json(rows)


@tool
def get_tracks_by_album(album_title: str) -> str:
    """List every track on an album, including its artist and album title."""
    log_tool_call("get_tracks_by_album")
    title = album_title.strip()
    if not title:
        return encode_json(
            {"error": "album_title must not be empty", "code": "INVALID_INPUT"}
        )

    matching = {"album": escaped_like(title)}
    rows, error = query_rows(
        """
        SELECT Track.TrackId AS track_id, Track.Name AS title,
               Artist.Name AS artist, Album.Title AS album,
               Album.AlbumId AS album_id
        FROM Track
        JOIN Album ON Album.AlbumId = Track.AlbumId
        JOIN Artist ON Artist.ArtistId = Album.ArtistId
        WHERE LOWER(Album.Title) LIKE LOWER(:album) ESCAPE '\\'
        ORDER BY Album.Title, Artist.Name, Track.TrackId
        """,
        matching,
    )
    if error:
        return error
    if not rows:
        suggestion, fuzzy_error = fuzzy_search_parameter(
            title, sql="SELECT Title AS title FROM Album", name="title"
        )
        if fuzzy_error:
            return fuzzy_error
        if suggestion and suggestion.casefold() != title.casefold():
            return get_tracks_by_album.func(suggestion)
        return not_found(f"No tracks found for album: {title}")

    return encode_json(
        {
            "tracks": rows,
            "total_count": len(rows),
            "truncated": False,
        }
    )


@tool
def get_tracks_by_artist(artist_name: str) -> str:
    """Find tracks by artist, returning a sample of at most 20 and the total count."""
    log_tool_call("get_tracks_by_artist")
    matching = {
        "artist": escaped_like(artist_name),
    }
    count_rows, error = query_rows(
        """
        SELECT COUNT(*) AS total_count
        FROM Track
        JOIN Album ON Album.AlbumId = Track.AlbumId
        JOIN Artist ON Artist.ArtistId = Album.ArtistId
        WHERE LOWER(Artist.Name) LIKE LOWER(:artist) ESCAPE '\\'
        """,
        matching,
    )
    if error:
        return error
    total_count = count_rows[0]["total_count"]
    if total_count == 0:
        suggestion, fuzzy_error = fuzzy_search_parameter(
            artist_name, sql="SELECT Name AS name FROM Artist", name="name"
        )
        if fuzzy_error:
            return fuzzy_error
        if suggestion and suggestion.casefold() != artist_name.casefold():
            return get_tracks_by_artist.func(suggestion)
        return not_found(f"No tracks found for artist: {artist_name}")

    rows, error = query_rows(
        """
        SELECT Track.TrackId AS track_id, Track.Name AS title,
               Artist.Name AS artist, Album.Title AS album
        FROM Track
        JOIN Album ON Album.AlbumId = Track.AlbumId
        JOIN Artist ON Artist.ArtistId = Album.ArtistId
        WHERE LOWER(Artist.Name) LIKE LOWER(:artist) ESCAPE '\\'
        ORDER BY Track.Name, Track.TrackId
        LIMIT 20
        """,
        matching,
    )
    if error:
        return error
    return encode_json(
        {
            "tracks": rows,
            "total_count": total_count,
            "truncated": total_count > len(rows),
        }
    )


@tool
def get_songs_by_genre(genre: str) -> str:
    """Browse up to 10 deterministic representative songs from a genre, one per artist."""
    log_tool_call("get_songs_by_genre")
    matching = {"genre": escaped_like(genre)}
    count_rows, error = query_rows(
        """
        SELECT COUNT(*) AS total_count,
               COUNT(DISTINCT Album.AlbumId) AS album_count,
               COUNT(DISTINCT Artist.ArtistId) AS artist_count
        FROM Track
        JOIN Album ON Album.AlbumId = Track.AlbumId
        JOIN Artist ON Artist.ArtistId = Album.ArtistId
        JOIN Genre ON Genre.GenreId = Track.GenreId
        WHERE LOWER(Genre.Name) LIKE LOWER(:genre) ESCAPE '\\'
        """,
        matching,
    )
    if error:
        return error
    total_count = count_rows[0]["total_count"]
    if total_count == 0:
        suggestion, fuzzy_error = fuzzy_search_parameter(
            genre, sql="SELECT Name AS name FROM Genre", name="name"
        )
        if fuzzy_error:
            return fuzzy_error
        if suggestion and suggestion.casefold() != genre.casefold():
            return get_songs_by_genre.func(suggestion)
        return not_found(f"No songs found for genre: {genre}")

    rows, error = query_rows(
        """
        WITH ranked_tracks AS (
            SELECT Track.TrackId AS track_id, Track.Name AS title,
                   Artist.Name AS artist, Album.Title AS album,
                   Genre.Name AS genre,
                   ROW_NUMBER() OVER (
                       PARTITION BY Artist.ArtistId
                       ORDER BY Track.TrackId
                   ) AS artist_track_rank
            FROM Track
            JOIN Album ON Album.AlbumId = Track.AlbumId
            JOIN Artist ON Artist.ArtistId = Album.ArtistId
            JOIN Genre ON Genre.GenreId = Track.GenreId
            WHERE LOWER(Genre.Name) LIKE LOWER(:genre) ESCAPE '\\'
        )
        SELECT track_id, title, artist, album, genre
        FROM ranked_tracks
        WHERE artist_track_rank = 1
        ORDER BY artist, track_id
        LIMIT 10
        """,
        matching,
    )
    if error:
        return error
    return encode_json(
        {
            "tracks": rows,
            "total_count": total_count,  # Number of matching tracks, not artists.
            "album_count": count_rows[0]["album_count"],
            "artist_count": count_rows[0]["artist_count"],
            "sample_size": len(rows),
            "truncated": total_count > len(rows),
        }
    )


@tool
def check_for_songs(title: str) -> str:
    """Search track titles using a fuzzy match, returning up to 10 results."""
    log_tool_call("check_for_songs")
    matching = {"title": escaped_like(title)}
    count_rows, error = query_rows(
        "SELECT COUNT(*) AS total_count FROM Track "
        "WHERE LOWER(Name) LIKE LOWER(:title) ESCAPE '\\'",
        matching,
    )
    if error:
        return error
    total_count = count_rows[0]["total_count"]
    if total_count == 0:
        suggestion, fuzzy_error = fuzzy_search_parameter(
            title, sql="SELECT Name AS name FROM Track", name="name"
        )
        if fuzzy_error:
            return fuzzy_error
        if suggestion and suggestion.casefold() != title.casefold():
            return check_for_songs.func(suggestion)
        return not_found(f"No songs found with title matching: {title}")

    rows, error = query_rows(
        """
        SELECT TrackId AS track_id, Name AS title
        FROM Track
        WHERE LOWER(Name) LIKE LOWER(:title) ESCAPE '\\'
        ORDER BY Name, TrackId
        LIMIT 10
        """,
        matching,
    )
    if error:
        return error
    return encode_json(
        {
            "tracks": rows,
            "total_count": total_count,
            "truncated": total_count > len(rows),
        }
    )


@tool
def get_track_details(track_id: str) -> str:
    """Get the complete catalog details for a track ID."""
    log_tool_call("get_track_details")
    numeric_id, error = validate_id(track_id, "track_id")
    if error:
        return error

    rows, error = query_rows(
        """
        SELECT Track.TrackId AS track_id, Track.Name AS title,
               Artist.Name AS artist, Album.Title AS album,
               Genre.Name AS genre, MediaType.Name AS media_type,
               Track.Composer AS composer, Track.Milliseconds AS milliseconds,
               Track.Bytes AS bytes, Track.UnitPrice AS unit_price
        FROM Track
        LEFT JOIN Album ON Album.AlbumId = Track.AlbumId
        LEFT JOIN Artist ON Artist.ArtistId = Album.ArtistId
        LEFT JOIN Genre ON Genre.GenreId = Track.GenreId
        JOIN MediaType ON MediaType.MediaTypeId = Track.MediaTypeId
        WHERE Track.TrackId = :track_id
        """,
        {"track_id": numeric_id},
    )
    if error:
        return error
    if not rows:
        return not_found(f"No track found with ID: {track_id}")
    return encode_json(rows[0])
