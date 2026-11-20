
from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo.

    Returns a list of results containing title, URL, and snippet.
    """
    if not query or not query.strip():
        return []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query.strip(),
                max_results=max_results,
            )

            return [
                {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                }
                for result in results
            ]

    except Exception as exc:
        raise RuntimeError(
            f"Web search failed: {exc}"
        ) from exc
