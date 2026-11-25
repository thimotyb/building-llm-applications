import requests
from bs4 import BeautifulSoup

def web_scrape(url: str) -> str:
    try:
        # Send explicit HTTP headers instead of the requests defaults.
        # Many sites, Wikipedia included, now reject the default
        # "python-requests/x.y" User-Agent with HTTP 403 Forbidden, so
        # web_scrape("https://en.wikipedia.org/wiki/...") would return an
        # error string instead of the page text. A browser-like User-Agent
        # plus an Accept-Language header gets the normal HTML page back.
        # This is the same header set used in ch04/web_scraping.py.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        # The timeout stops one slow or unresponsive site from blocking the
        # whole research graph: after 15 seconds requests raises an
        # exception, which the except branch below turns into an error string.
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)

            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"
