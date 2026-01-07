"""Trading universe - top 100 liquid stocks and ETFs.

This list includes:
- Major ETFs (indices, sectors)
- Mega-cap tech
- Financials
- Healthcare
- Consumer
- Energy
- Industrials
- Other highly liquid names

All selected for high liquidity and options availability.
"""

# Major Index ETFs
INDEX_ETFS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "DIA",   # Dow Jones
    "VTI",   # Total Stock Market
]

# Sector ETFs
SECTOR_ETFS = [
    "XLF",   # Financials
    "XLK",   # Technology
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
]

# Volatility & Bonds
VOLATILITY_BONDS = [
    "TLT",   # 20+ Year Treasury
    "GLD",   # Gold
    "SLV",   # Silver
    "USO",   # Oil
]

# Mega Cap Tech
MEGA_CAP_TECH = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "META",  # Meta
    "NVDA",  # NVIDIA
    "TSLA",  # Tesla
    "AMD",   # AMD
    "INTC",  # Intel
    "CRM",   # Salesforce
    "ORCL",  # Oracle
    "ADBE",  # Adobe
    "NFLX",  # Netflix
    "AVGO",  # Broadcom
    "CSCO",  # Cisco
]

# Financials
FINANCIALS = [
    "JPM",   # JPMorgan
    "BAC",   # Bank of America
    "WFC",   # Wells Fargo
    "GS",    # Goldman Sachs
    "MS",    # Morgan Stanley
    "C",     # Citigroup
    "BLK",   # BlackRock
    "SCHW",  # Schwab
    "AXP",   # American Express
    "V",     # Visa
    "MA",    # Mastercard
    "PYPL",  # PayPal
]

# Healthcare
HEALTHCARE = [
    "JNJ",   # Johnson & Johnson
    "UNH",   # UnitedHealth
    "PFE",   # Pfizer
    "MRK",   # Merck
    "ABBV",  # AbbVie
    "LLY",   # Eli Lilly
    "TMO",   # Thermo Fisher
    "BMY",   # Bristol-Myers
    "AMGN",  # Amgen
    "GILD",  # Gilead
]

# Consumer
CONSUMER = [
    "WMT",   # Walmart
    "COST",  # Costco
    "HD",    # Home Depot
    "LOW",   # Lowe's
    "TGT",   # Target
    "NKE",   # Nike
    "SBUX",  # Starbucks
    "MCD",   # McDonald's
    "KO",    # Coca-Cola
    "PEP",   # PepsiCo
    "PG",    # Procter & Gamble
]

# Energy
ENERGY = [
    "XOM",   # Exxon
    "CVX",   # Chevron
    "COP",   # ConocoPhillips
    "SLB",   # Schlumberger
    "EOG",   # EOG Resources
]

# Industrials
INDUSTRIALS = [
    "CAT",   # Caterpillar
    "BA",    # Boeing
    "HON",   # Honeywell
    "UPS",   # UPS
    "RTX",   # Raytheon
    "DE",    # Deere
    "LMT",   # Lockheed Martin
    "GE",    # GE Aerospace
    "MMM",   # 3M
]

# Communications
COMMUNICATIONS = [
    "DIS",   # Disney
    "CMCSA", # Comcast
    "VZ",    # Verizon
    "T",     # AT&T
    "TMUS",  # T-Mobile
]

# Other Large Caps
OTHER = [
    "BRK.B", # Berkshire
    "UNP",   # Union Pacific
    "UBER",  # Uber
    "ABNB",  # Airbnb
    "SQ",    # Block
    "COIN",  # Coinbase
    "SHOP",  # Shopify
    "NOW",   # ServiceNow
    "SNOW",  # Snowflake
    "PLTR",  # Palantir
]

# Combined universe (exactly 100)
UNIVERSE_100 = (
    INDEX_ETFS +      # 5
    SECTOR_ETFS +     # 10
    VOLATILITY_BONDS + # 4
    MEGA_CAP_TECH +   # 15
    FINANCIALS +      # 12
    HEALTHCARE +      # 10
    CONSUMER +        # 11
    ENERGY +          # 5
    INDUSTRIALS +     # 9
    COMMUNICATIONS +  # 5
    OTHER             # 10
)  # Total: 96, add 4 more

# Add a few more liquid names to reach 100
UNIVERSE_100 = UNIVERSE_100 + [
    "F",     # Ford
    "GM",    # GM
    "AAL",   # American Airlines
    "DAL",   # Delta
]

assert len(UNIVERSE_100) == 100, f"Universe has {len(UNIVERSE_100)} stocks, expected 100"


def get_universe(size: int = 100) -> list[str]:
    """Get trading universe.

    Args:
        size: Number of stocks (10, 25, 50, or 100)

    Returns:
        List of ticker symbols
    """
    if size == 10:
        # Core ETFs + mega caps
        return ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"]
    elif size == 25:
        return UNIVERSE_100[:25]
    elif size == 50:
        return UNIVERSE_100[:50]
    else:
        return UNIVERSE_100


def get_sectors() -> dict[str, list[str]]:
    """Get stocks grouped by sector."""
    return {
        "index_etfs": INDEX_ETFS,
        "sector_etfs": SECTOR_ETFS,
        "volatility_bonds": VOLATILITY_BONDS,
        "mega_cap_tech": MEGA_CAP_TECH,
        "financials": FINANCIALS,
        "healthcare": HEALTHCARE,
        "consumer": CONSUMER,
        "energy": ENERGY,
        "industrials": INDUSTRIALS,
        "communications": COMMUNICATIONS,
        "other": OTHER,
    }
