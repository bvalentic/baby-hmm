# Baby's first "trading algorithm"

# originally basic_algo
# begin at start_date with initial funds
# buy X amount (5%? 25%?) if bullish in that interval
# sell Y amount if bearish
# add to funding pool every week
def signal_trader(data_frame, initial_funds = 1000, shares = 0.0, buy_percentage = 0.25, sell_percentage = 0.50):
    cash_on_hand = initial_funds # assume USD
    portfolio_history = [] 

    for i in range(len(data_frame)):
        # ensure prices are scalars, not Series
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

        # valuation at the end of the day
        total_value = float(cash_on_hand + (shares * close_price))
        portfolio_history.append(total_value)
    
    return portfolio_history

# the old "HMM Strategy"
def hmm_pure_strategy(data_frame):
    # we shift signal by 1 because we trade at the close based on today's state for tomorrow
    data_frame['Strategy_Returns'] = data_frame['Signal'].shift(1) * data_frame['Returns']
    return data_frame['Strategy_Returns']

# simple buy-and-hold algorithm
def buy_and_hold_pure_strategy(data_frame, initial_funds = 1000, shares = 0.0):
    cash_on_hand = initial_funds # assume USD
    portfolio_history = []

    # buy full amount of asset on day 0, then hold
    day_0_price = data_frame['Open'].iloc[0]
    if hasattr(day_0_price, 'item'): day_0_price = day_0_price.item()

    dollar_to_spend = cash_on_hand
    shares_to_buy = dollar_to_spend / day_0_price
    cash_on_hand -= dollar_to_spend
    shares += shares_to_buy

    for i in range(len(data_frame)):
        # no buy/sell logic, just grabbing daily returns
        close_price = data_frame['Close'].iloc[i]
        if hasattr(close_price, 'item'): close_price = close_price.item()

        # valuation at the end of the day
        total_value = float(cash_on_hand + (shares * close_price))
        portfolio_history.append(total_value)

    return portfolio_history
