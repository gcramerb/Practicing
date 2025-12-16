import requests

def get_data(url):
    response = requests.get(url)
    data = bytes.decode(response.content)
    data = data.split("x-coordinate")[1]
    return response


if __name__=="__main__":
    URL = "https://docs.google.com/document/u/0/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub?pli=1"
    res = get_data(URL)