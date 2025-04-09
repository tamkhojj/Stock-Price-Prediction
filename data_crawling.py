import requests
from bs4 import BeautifulSoup
import json
import os


def get_stock_list(url="https://www.cophieu68.vn/market/markets.php?id=^vnindex"):
    """
    Retrieve a list of stock symbols and company names from cophieu68.vn.

    Args:
        url (str): The URL to fetch the stock list from.

    Returns:
        list: A list of tuples containing (stock symbol, company name).
    """
    response = requests.get(url)
    response.encoding = "utf-8"


    soup = BeautifulSoup(response.text, "html.parser")
    stock_elements = soup.find_all("a", href=True)

    stock_data = set()

    for stock in stock_elements:
        stock_code_tag = stock.find("div", style="text-transform:uppercase; font-weight:bold; font-size:18px")
        stock_code = stock_code_tag.text.strip().upper() if stock_code_tag else None

        company_name_tag = stock.find("div", class_="mobile90")
        company_name = company_name_tag.text.strip() if company_name_tag else None

        if stock_code:
            stock_data.add((stock_code, company_name))

    return list(stock_data)


def get_stock_history(stock_code, max_pages=20):
    """
    Retrieve historical stock data for a given stock symbol.

    Args:
        stock_code (str): The stock symbol to retrieve data for.
        max_pages (int): Maximum number of pages to crawl.

    Returns:
        list: A list of dictionaries containing stock history data.
    """
    all_data = []

    for page in range(1, max_pages + 1):
        url = f"https://www.cophieu68.vn/quote/history.php?cP={page}&id={stock_code}"

        response = requests.get(url)
        response.encoding = "utf-8"


        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", id="history")

        rows = table.find_all("tr")[1:]  # Skip header row

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 6:
                date = cols[0].text.strip()
                close_price = cols[1].find("strong").text.strip() if cols[1].find("strong") else cols[1].text.strip()
                volume = cols[2].text.strip()
                open_price = cols[3].text.strip()
                high_price = cols[4].text.strip()
                low_price = cols[5].text.strip()

                data_entry = {
                    "Date": date,
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume
                }

                all_data.append(data_entry)

    return all_data


def save_stock_data():
    """
    Retrieve stock codes, crawl historical data (20 pages max per stock),
    and save each stock's data into a separate JSON file.
    """
    stocks = get_stock_list()
    os.makedirs("stock_data", exist_ok=True)

    for stock_code, _ in stocks:
        stock_data = get_stock_history(stock_code, max_pages=20)

        if stock_data:
            filename = f"stock_data/{stock_code}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(stock_data, f, ensure_ascii=False, indent=4)
        else:
            print(f"No data found")


if __name__ == "__main__":
    save_stock_data()
