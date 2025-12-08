from server import fetch_stock_data, generate_weather_widget

print("Testing fetch_stock_data('AAPL')...")
try:
    stock_data = fetch_stock_data("AAPL")
    print(f"Stock Data: {stock_data[:200]}...") # Print first 200 chars
except Exception as e:
    print(f"Stock Data Error: {e}")

print("\nTesting generate_weather_widget('London', 'light')...")
try:
    weather_widget = generate_weather_widget("London", "light")
    with open("weather_widget_light.html", "w", encoding="utf-8") as f:
        f.write(weather_widget)
    print(f"Weather Widget (Light): {weather_widget[:200]}...")
except Exception as e:
    print(f"Weather Widget Error: {e}")

print("\nTesting generate_weather_widget('Tokyo', 'dark')...")
try:
    weather_widget_dark = generate_weather_widget("Tokyo", "dark")
    with open("weather_widget_dark.html", "w", encoding="utf-8") as f:
        f.write(weather_widget_dark)
    print(f"Weather Widget (Dark): {weather_widget_dark[:200]}...")
except Exception as e:
    print(f"Weather Widget Error: {e}")
