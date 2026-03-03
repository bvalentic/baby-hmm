# Baby's first "trading algorithm"

# begin at start_date with initial funds
# buy X amount (5%?) if bullish in that interval
# sell Y amount (5%?) if bearish
# add to funding pool every week
def algorithm(data_frame):
    initial_funds = 1000.0 # USD
    cash_on_hand = initial_funds
    shares = 0.0
    portfolio_history = []

    buy_percentage = 0.05  # Use 5% of AVAILABLE CASH to buy
    sell_percentage = 0.05 # Sell 5% of CURRENT SHARES held

    for i in range(len(data_frame)):
        # 1. Ensure prices are scalars, not Series
        # .item() extracts the single value from a Series
        open_price = data_frame['Open'].iloc[i]
        if hasattr(open_price, 'item'): open_price = open_price.item()
        close_price = data_frame['Close'].iloc[i]
        if hasattr(close_price, 'item'): close_price = close_price.item()
        signal = data_frame['Signal'].iloc[i]
        if hasattr(signal, 'item'): signal = signal.item()

        if signal == 1:
            # BUY LOGIC
            dollar_to_spend = cash_on_hand * buy_percentage
            shares_to_buy = dollar_to_spend / open_price
            
            cash_on_hand -= dollar_to_spend
            shares += shares_to_buy
        else:
            # SELL LOGIC
            shares_to_sell = shares * sell_percentage
            cash_received = shares_to_sell * open_price
            
            shares -= shares_to_sell
            cash_on_hand += cash_received

        # Valuation at the end of the day
        total_value = float(cash_on_hand + (shares * close_price))
        portfolio_history.append(total_value)
    
    return portfolio_history
