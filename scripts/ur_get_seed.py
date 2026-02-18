from playwright.sync_api import sync_playwright

QUERY = "ggmbh"  # change later to eg / gnr strategy

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Start page
    page.goto("https://www.unternehmensregister.de/de", wait_until="networkidle")

    # Go to search page (site uses routing; this is the page you shared)
    page.goto("https://www.unternehmensregister.de/de/suche", wait_until="networkidle")

    # Fill in company name/search term fields (labels vary; use placeholder/name heuristics)
    # This works on the current UI: it targets the input that accepts the company name.
    page.get_by_role("textbox").first.fill(QUERY)

    # Submit search (press Enter works reliably)
    page.get_by_role("textbox").first.press("Enter")
    page.wait_for_load_state("networkidle")

    # Switch to "Registerinformationen" view if there is a UI tab/button
    # If the tab exists, click it; otherwise we just print current URL.
    try:
        page.get_by_text("Registerinformationen").click()
        page.wait_for_load_state("networkidle")
    except Exception:
        pass

    print(page.url)
    browser.close()
