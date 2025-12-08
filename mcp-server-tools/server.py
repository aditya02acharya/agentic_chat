from fastmcp import FastMCP
import yfinance as yf
import requests
from typing import Literal

# Initialize FastMCP server
mcp = FastMCP("mcp-server-tools")

def fetch_stock_data(ticker: str) -> str:
    """
    Retrieves stock data for a given ticker symbol using yfinance.
    Returns a JSON string with key stock information.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant data
        data = {
            "symbol": info.get("symbol", ticker),
            "shortName": info.get("shortName"),
            "currentPrice": info.get("currentPrice"),
            "currency": info.get("currency"),
            "dayHigh": info.get("dayHigh"),
            "dayLow": info.get("dayLow"),
            "volume": info.get("volume"),
            "marketCap": info.get("marketCap"),
            "sector": info.get("sector"),
            "website": info.get("website"),
            "longBusinessSummary": info.get("longBusinessSummary")
        }
        return str(data)
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {str(e)}"

def generate_weather_widget(city: str, mode: Literal["light", "dark"] = "light") -> str:
    """
    Generates an HTML/CSS/JS widget for the current weather in a specified city.
    Supports light and dark modes.
    Returns an HTML string.
    """
    try:
        # 1. Geocoding to get lat/lon
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json()
        
        if not geo_res.get("results"):
            return f"Could not find location: {city}"
            
        location = geo_res["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        city_name = location["name"]
        country = location.get("country", "")

        # 2. Fetch Weather Data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,is_day&timezone=auto"
        weather_res = requests.get(weather_url).json()
        
        current = weather_res.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        weather_code = current.get("weather_code", 0)
        is_day = current.get("is_day", 1)
        
        # Map WMO weather codes to conditions
        condition = "sunny" # default
        if weather_code in [0, 1]:
            condition = "sunny" if is_day else "clear-night"
        elif weather_code in [2, 3, 45, 48]:
            condition = "overcast"
        elif weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
            condition = "rain"
        elif weather_code in [71, 73, 75, 77, 85, 86]:
            condition = "snow"
        elif weather_code in [95, 96, 99]:
            condition = "thunderstorm"

        # 3. Generate HTML Widget
        bg_color = "#ffffff" if mode == "light" else "#1e1e1e"
        text_color = "#333333" if mode == "light" else "#ffffff"
        
        # CSS Animations based on condition
        animation_css = ""
        weather_icon = ""
        weather_div = ""
        
        if condition == "sunny":
            weather_icon = "☀️"
            animation_css = """
            .sun {
                width: 50px;
                height: 50px;
                background: #FFD700;
                border-radius: 50%;
                box-shadow: 0 0 20px #FFD700;
                animation: spin 10s linear infinite;
            }
            @keyframes spin { 100% { transform: rotate(360deg); } }
            """
            weather_div = '<div class="sun"></div>'
        elif condition == "clear-night":
            weather_icon = "🌙"
            weather_div = '<div style="font-size: 40px;">🌙</div>'
        elif condition == "rain":
            weather_icon = "🌧️"
            animation_css = """
            .rain-drop {
                width: 2px;
                height: 10px;
                background: #00BFFF;
                position: absolute;
                animation: fall 1s linear infinite;
            }
            @keyframes fall { to { transform: translateY(50px); opacity: 0; } }
            """
            weather_div = """
            <div style="position: relative; width: 50px; height: 50px; overflow: hidden;">
                <div class="rain-drop" style="left: 10px; animation-delay: 0s;"></div>
                <div class="rain-drop" style="left: 20px; animation-delay: 0.2s;"></div>
                <div class="rain-drop" style="left: 30px; animation-delay: 0.4s;"></div>
                <div class="rain-drop" style="left: 40px; animation-delay: 0.1s;"></div>
                🌧️
            </div>
            """
        elif condition == "overcast":
            weather_icon = "☁️"
            animation_css = """
            .cloud {
                animation: float 3s ease-in-out infinite;
            }
            @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
            """
            weather_div = '<div class="cloud" style="font-size: 40px;">☁️</div>'
        else:
            weather_icon = "🌡️"
            weather_div = f'<div style="font-size: 40px;">{weather_icon}</div>'

        html = f"""
        <div style="
            font-family: 'Segoe UI', sans-serif;
            background-color: {bg_color};
            color: {text_color};
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            width: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        ">
            <style>
                {animation_css}
            </style>
            <div style="font-size: 1.2em; font-weight: bold;">{city_name}, {country}</div>
            <div style="display: flex; align-items: center; gap: 15px;">
                {weather_div}
                <div style="font-size: 2.5em; font-weight: bold;">{temp}°C</div>
            </div>
            <div style="text-transform: capitalize; opacity: 0.8;">{condition.replace('-', ' ')}</div>
        </div>
        """
        return html

    except Exception as e:
        return f"Error generating weather widget: {str(e)}"

@mcp.tool()
def get_stock_data(ticker: str) -> str:
    """
    Retrieves stock data for a given ticker symbol using yfinance.
    Returns a JSON string with key stock information.
    """
    return fetch_stock_data(ticker)

@mcp.tool()
def get_weather_widget(city: str, mode: Literal["light", "dark"] = "light") -> str:
    """
    Generates an HTML/CSS/JS widget for the current weather in a specified city.
    Supports light and dark modes.
    Returns an HTML string.
    """
    return generate_weather_widget(city, mode)

if __name__ == "__main__":
    mcp.run()
