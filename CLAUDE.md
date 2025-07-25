# CLAUDE.md - SqueezeFlow Trader 2

Dieses Dokument stellt eine vollständige Anleitung für Claude Code zur Arbeit mit der SqueezeFlow Trader 2 Codebase dar. Es definiert Arbeitsregeln, Systemarchitektur und technische Spezifikationen.

## 🚨 ARBEITSREGELN FÜR CLAUDE

### Grundprinzipien
- ❌ **KEINE einfachen/simplen Wege** - Implementierungen müssen vollständig und robust sein
- ✅ **Genau wie gewünscht** - Exakte Umsetzung der Anforderungen ohne Abkürzungen
- ✅ **Gegenfragen stellen** - Bei Unklarheiten immer nachfragen vor Implementierung
- ✅ **Vollständiges Verständnis zeigen** - Detaillierte Analyse vor jeder Aktion
- ✅ **Go-Signal abwarten** - Erst nach expliziter Freigabe mit Implementierung beginnen
- ✅ **Alles sauber prüfen** - Tests, Validierung und Verifikation sind Pflicht
- ✅ **Vollständige Dokumentation** - Alle Änderungen müssen dokumentiert werden

### Implementierungsflow
1. **Anforderung verstehen** und detailliert analysieren
2. **Fragen stellen** bei unklaren Punkten
3. **Kompletten Plan präsentieren** mit allen technischen Details
4. **Go-Signal abwarten** vom Benutzer
5. **Implementierung durchführen** mit vollständiger Prüfung
6. **Tests und Validierung** durchführen
7. **Dokumentation aktualisieren**

## 📋 PROJEKTÜBERSICHT

SqueezeFlow Trader 2 ist ein hochentwickeltes Cryptocurrency Trading System basierend auf **Squeeze Detection** durch CVD-Divergenz-Analyse zwischen Spot- und Futures-Märkten.

### Kernkonzept: Squeeze Detection
Das System erkennt "Squeeze"-Situationen durch Analyse von:
- **Long Squeeze**: Preis↑ + Spot CVD↑ + Futures CVD↓ → Negative Score
- **Short Squeeze**: Preis↓ + Spot CVD↓ + Futures CVD↑ → Positive Score
- **CVD Divergenz**: Unterschiede zwischen Spot- und Futures-Cumulative Volume Delta

## 🏗️ SYSTEMARCHITEKTUR

### Docker-basierte Mikroservice-Architektur
```yaml
Services (docker-compose.yml):
├── aggr-influx (InfluxDB 1.8.10) - Zeitreihen-Datenbank
├── aggr-server (Node.js) - Echtzeit-Datensammlung von 20+ Exchanges
├── redis (7-alpine) - Caching und Message Queue  
├── grafana - Monitoring und Dashboards
├── squeezeflow-calculator - Core Signal Generator Service
├── oi-tracker - Open Interest Tracking Service
├── freqtrade - Trading Engine mit FreqAI Integration
├── freqtrade-ui - Web Interface für Trading Management
└── system-monitor - System Health Monitoring Service
```

### Network-Architektur
```yaml
Networks:
├── squeezeflow_network - Interne Service-Kommunikation
└── aggr_backend - External aggr-server Integration
```

### Datenfluss-Pipeline (Aktualisiert 2025)
```
Exchange APIs → aggr-server → InfluxDB → Symbol/Market/OI Discovery → SqueezeFlow Calculator → Redis → Freqtrade → Order Execution
                    ↓              ↓                ↓                          ↓               ↓
                Grafana ←─────── System Monitor ←──────────────────────────────────────────────────
```

### Neue Discovery-Services-Architektur
```
InfluxDB (Source of Truth)
    ↓
Symbol Discovery: Welche base symbols haben Daten?
    ↓ 
Market Discovery: Welche markets pro symbol?
    ↓
OI Discovery: Welche OI-symbols pro base symbol?
    ↓
Services (SqueezeFlow Calculator, FreqTrade) - Vollautomatisch
```

## 🔍 DISCOVERY-SERVICES (NEU 2025)

### Robuste Symbol/Market/OI-Discovery
Das System verwendet **datengetriebene Discovery** statt hardcoded Listen für maximale Robustheit:

#### Symbol Discovery (`utils/symbol_discovery.py`)
```python
# Automatische Erkennung verfügbarer Symbols aus InfluxDB
active_symbols = symbol_discovery.discover_symbols_from_database(
    min_data_points=500,  # Qualitätsschwelle
    hours_lookback=24     # Zeitraum-Validierung
)
# Ergebnis: ['BTC', 'ETH'] - nur Symbols mit echten Daten
```

#### Market Discovery (`utils/market_discovery.py`)  
```python
# Findet echte Markets pro Symbol aus DB
markets = market_discovery.get_markets_by_type('BTC')
# Ergebnis: {'spot': ['BINANCE:btcusdt', ...], 'perp': ['BINANCE_FUTURES:btcusdt', ...]}
```

#### OI Discovery (Open Interest)
```python
# Findet verfügbare OI-Symbols pro base symbol
oi_symbols = symbol_discovery.discover_oi_symbols_for_base('BTC')
# Ergebnis: ['BTCUSDT', 'BTCUSD'] - echte OI-Daten aus DB
```

### Vorteile der Discovery-Architektur
- ✅ **Keine hardcoded Symbol-Listen** mehr
- ✅ **Automatische Skalierung** für neue Symbols/Markets
- ✅ **Datenqualitäts-Validierung** integriert
- ✅ **Robuste Fallback-Mechanismen**
- ✅ **Multi-Exchange-Support** automatisch
- ✅ **FreqTrade Multi-Pair-Support** vollautomatisch

## ⚙️ KONFIGURATIONSSYSTEM

### Hierarchische Konfigurationsstruktur
```/exit
config/
├── config.yaml - Hauptsystemkonfiguration
├── exchanges.yaml - Exchange API Settings & Credentials
├── risk_management.yaml - Risikomanagement Parameter
├── execution_config.yaml - Order Execution Settings
├── ml_config.yaml - Machine Learning Konfiguration
├── trading_parameters.yaml - Trading-Strategieparameter
└── feature_toggles.yaml - Feature-Flags & Toggles
```

### Umgebungsmodi
- **Development**: `python init.py --mode development` (Localhost, Debug-Logging)
- **Production**: `python init.py --mode production` (Live-Trading, Optimiert)
- **Docker**: `python init.py --mode docker` (Vollständig containerisiert)

## 🗄️ DATENBANKSTRUKTUREN (VOLLSTÄNDIG)

### InfluxDB Measurements (Zeitreihen-Datenbank)

#### 1. squeeze_signals
```python
measurement: "squeeze_signals"
tags: {
    symbol: str,           # z.B. "BTCUSDT"
    exchange: str,         # z.B. "binance"
    signal_type: str       # "LONG_SQUEEZE", "SHORT_SQUEEZE", "NEUTRAL"
}
fields: {
    squeeze_score: float,      # -1.0 bis +1.0
    price_change: float,       # Prozentuale Preisänderung
    volume_surge: float,       # Volume-Multiplikator
    oi_change: float,          # Open Interest Änderung
    cvd_divergence: float      # CVD-Divergenz-Stärke
}
```

#### 2. positions  
```python
measurement: "positions"
tags: {
    symbol: str,               # Trading Pair
    exchange: str,             # Exchange Name
    side: str,                 # "buy", "sell"
    status: str,               # "open", "closed", "cancelled"
    strategy_name: str,        # "SqueezeFlowFreqAI"
    is_dry_run: bool          # true/false
}
fields: {
    position_id: str,          # Eindeutige Position ID
    entry_price: float,        # Entry-Preis
    size: float,               # Position-Größe
    fees: float,               # Gesamte Gebühren
    exit_price: float,         # Exit-Preis (null wenn offen)
    stop_loss: float,          # Stop-Loss-Preis
    take_profit: float,        # Take-Profit-Preis
    pnl: float,                # Profit/Loss absolut
    pnl_percentage: float      # Profit/Loss in Prozent
}
```

#### 3. trades
```python
measurement: "trades"  
tags: {
    symbol: str,               # Trading Pair
    exchange: str,             # Exchange Name
    side: str,                 # "buy", "sell"
    order_type: str,           # "market", "limit"
    is_dry_run: bool          # true/false
}
fields: {
    trade_id: str,             # Eindeutige Trade ID
    position_id: str,          # Referenz zur Position
    price: float,              # Execution Price
    amount: float,             # Trade Amount
    fee: float,                # Trade Fee
    order_id: str              # Exchange Order ID
}
```

#### 4. ml_predictions
```python
measurement: "ml_predictions"
tags: {
    symbol: str,               # Trading Pair
    model_name: str           # "LightGBMRegressorMultiTarget"
}
fields: {
    prediction: float,         # ML Prediction Value
    confidence: float,         # Prediction Confidence 0-1
    feature_importance: str    # JSON der Feature-Wichtigkeiten
}
```

#### 5. system_metrics
```python
measurement: "system_metrics"
tags: {
    metric_name: str,          # "cpu_usage", "memory_usage", etc.
    component: str,            # "freqtrade", "aggr-server", etc.
    metric_unit: str          # "percent", "bytes", "count"
}
fields: {
    metric_value: float        # Metric Value
}
```

### Aggr-Server InfluxDB Schema (Node.js)

#### trades_{timeframe} (z.B. trades_1m, trades_5m)
```javascript
measurement: 'trades_' + timeframe  // z.B. 'trades_1m'
tags: {
    market: str                     // z.B. "BINANCE:btcusdt"
}
fields: {
    vbuy: float,                   // Buy Volume
    vsell: float,                  // Sell Volume  
    cbuy: int,                     // Buy Count
    csell: int,                    // Sell Count
    lbuy: float,                   // Liquidation Buy Volume
    lsell: float,                  // Liquidation Sell Volume
    open: float,                   // OHLC Open Price
    high: float,                   // OHLC High Price
    low: float,                    // OHLC Low Price
    close: float                   // OHLC Close Price
}
```

### SQLite Datenbank (Freqtrade)

#### trades (Haupttabelle - 48 Felder)
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    exchange VARCHAR NOT NULL,
    pair VARCHAR NOT NULL,
    base_currency VARCHAR,
    stake_currency VARCHAR,
    is_open BOOLEAN NOT NULL DEFAULT 1,
    fee_open FLOAT DEFAULT 0.0,
    fee_close FLOAT DEFAULT 0.0,
    open_rate FLOAT,
    close_rate FLOAT,
    realized_profit FLOAT DEFAULT 0.0,
    stake_amount FLOAT NOT NULL,
    amount FLOAT,
    open_date DATETIME NOT NULL,
    close_date DATETIME,
    stop_loss FLOAT,
    is_stop_loss_trailing BOOLEAN DEFAULT 0,
    max_rate FLOAT,
    min_rate FLOAT,
    exit_reason VARCHAR,
    strategy VARCHAR,
    enter_tag VARCHAR,
    leverage FLOAT DEFAULT 1.0,
    is_short BOOLEAN DEFAULT 0
    -- + 24 weitere Felder für erweiterte Trade-Daten
);
```

#### orders (26 Felder)
```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    ft_trade_id INTEGER REFERENCES trades(id),
    ft_order_side VARCHAR NOT NULL,
    ft_pair VARCHAR NOT NULL,
    ft_is_open BOOLEAN NOT NULL DEFAULT 1,
    order_id VARCHAR NOT NULL,
    status VARCHAR,
    price FLOAT,
    amount FLOAT,
    filled FLOAT DEFAULT 0,
    cost FLOAT
    -- + 15 weitere Felder
);
```

### Redis Cache-Strukturen

#### Squeeze Signal Cache
```python
key_pattern: "squeeze_signal:{symbol}:{lookback}"  # z.B. "squeeze_signal:BTCUSDT:60"
data_structure: {
    'squeeze_score': float,        # -1.0 bis +1.0
    'signal_type': str,           # Klassifikation
    'signal_strength': float,     # Absoluter Score
    'timestamp': str,             # ISO DateTime
    'components': {
        'price_factor': float,
        'divergence_factor': float,
        'trend_factor': float
    }
}
ttl: 300  # 5 Minuten Cache-Lebenszeit
```

## 📁 WICHTIGE DATEISTRUKTUR-UPDATES (2025)

### Neue Core-Dateien
```
utils/
├── symbol_discovery.py      # Automatische Symbol-Erkennung aus DB
├── market_discovery.py      # Robuste Market-Discovery  
├── cvd_analysis_tool.py     # Flexible CVD-Analyse für alle Symbols
└── exchange_mapper.py       # Bereinigt - nur noch Klassifikation

services/
└── squeezeflow_calculator.py # Vollautomatische Symbol-Discovery

freqtrade/user_data/strategies/
└── SqueezeFlowFreqAI.py      # Robuste Pair-Konvertierung + OI-Discovery

backtest/
└── engine.py                # CVD-Tool-Schema für Market-Discovery

main.py                      # Docker-ready, vereinfacht, nur funktionierende Components
```

### Entfernte/Archivierte Dateien
```
Archived/
├── main.py                  # Alte complex main.py (20+ nicht-existente Imports)
├── fix_grafana_influx_queries.py # One-time fix script
└── test_grafana_queries.sql # Test script

# Entfernt aus exchange_mapper.py:
- _generate_markets_for_symbol() # 107 Zeilen Market-Generierung entfernt
- market_templates              # Hardcoded Templates entfernt  
```

## 🧮 SQUEEZE-ALGORITHMUS (EXAKTE PARAMETER)

### CVD-BERECHNUNGS-METHODOLOGIE (VERIFIZIERT 2025)

#### Industrie-Standard CVD Berechnung
Nach umfassender Recherche und Verifikation gegen professionelle Plattformen (aggr.trade, Velo Data) wurde die korrekte CVD-Berechnung implementiert:

```python
# VERIFIZIERTE CVD-FORMEL (Industrie-Standard):
# Step 1: Berechne per-Minute Volume Delta
volume_delta = vbuy - vsell  # Buy Volume minus Sell Volume

# Step 2: Berechne CUMULATIVE Volume Delta (laufende Summe)
cvd = volume_delta.cumsum()  # Running total über Zeit
```

**Kritische Erkenntnis**: CVD ist NICHT das per-Minute Delta, sondern die **kumulative Summe** aller Volume-Deltas über die Zeit. Dies entspricht dem Industrie-Standard und wurde durch reale Marktdaten verifiziert.

#### CVD-Implementierung in allen Systemkomponenten

**1. SqueezeFlow Calculator Service (services/squeezeflow_calculator.py:194-196)**
```python
# Step 1: Calculate per-minute volume delta (Buy Volume - Sell Volume)
spot_df['total_cvd_spot'] = spot_df['total_vbuy_spot'] - spot_df['total_vsell_spot']
spot_df = spot_df.set_index('time').sort_index()
# Step 2: Calculate CUMULATIVE Volume Delta (running total) - EXACT same logic as debug tool
spot_df['total_cvd_spot_cumulative'] = spot_df['total_cvd_spot'].cumsum()
spot_cvd = spot_df['total_cvd_spot_cumulative']  # Use cumulative CVD, not per-minute delta
```

**2. Backtest Engine (backtest/engine.py:193-198)**
```python
# Step 1: Calculate per-minute volume delta (Buy Volume - Sell Volume)
spot_df['total_cvd_spot'] = spot_df['total_vbuy_spot'] - spot_df['total_vsell_spot']
spot_df = spot_df.set_index('time').sort_index()
# Step 2: Calculate CUMULATIVE Volume Delta (running total) - EXACT same logic as debug tool
spot_df['total_cvd_spot_cumulative'] = spot_df['total_cvd_spot'].cumsum()
spot_cvd = spot_df['total_cvd_spot_cumulative']  # Use cumulative CVD, not per-minute delta
```

**3. Debug Tool Verifikation (btc_cvd_debug_tool.py)**
```python
# Step 1: Calculate per-minute volume delta
spot_df['spot_cvd'] = spot_df['vbuy'] - spot_df['vsell']
# Step 2: Calculate CUMULATIVE Volume Delta (running total)
spot_df['spot_cvd_cumulative'] = spot_df['spot_cvd'].cumsum()
```

#### Reale Marktdaten-Verifikation (Juli 23, 2025)
- **SPOT CVD**: -271 Millionen USD (massiver Verkaufsdruck)
- **FUTURES CVD**: -1,122 Millionen USD (extreme Futures-Verkäufe)
- **CVD-Divergenz**: 851 Millionen USD Unterschied zwischen Märkten
- **Datenqualität**: 47 SPOT + 16 PERP Exchanges, vollständige Marktabdeckung

#### Exchange-Klassifikation (exchange_mapper.py)
```python
# BTC Markets (Vollständige Klassifikation)
BTC_SPOT_MARKETS = [47 Exchanges]    # BINANCE:btcusdt, COINBASE:BTC-USD, etc.
BTC_PERP_MARKETS = [16 Exchanges]    # BINANCE_FUTURES:btcusdt, BYBIT:BTCUSDT, etc.

# ETH Markets (Vollständige Klassifikation)  
ETH_SPOT_MARKETS = [41 Exchanges]    # BINANCE:ethusdt, COINBASE:ETH-USD, etc.
ETH_PERP_MARKETS = [15 Exchanges]    # BINANCE_FUTURES:ethusdt, BYBIT:ETHUSDT, etc.
```

#### Systemweite CVD-Updates (Juli 24, 2025)
Alle Systemkomponenten wurden mit der verifizierten CVD-Methodologie aktualisiert:

1. **Live Calculator**: ✅ services/squeezeflow_calculator.py - Updated mit cumsum()
2. **Backtest Engine**: ✅ backtest/engine.py - Updated mit identical methodology  
3. **FreqTrade Strategy**: ✅ Bereits korrekt (nutzt Redis signals)
4. **Exchange Mapper**: ✅ Komplette aggr-server Mappings integriert

**Resultat**: Alle Systemkomponenten verwenden nun identische, industrie-verifizierte CVD-Berechnung.

### Squeeze Score Berechnung (squeeze_score_calculator.py)

#### Grundgewichtungen
```python
# Zeile 24-28: Constructor Defaults
price_weight: float = 0.3          # 30% Preis-Komponente
spot_cvd_weight: float = 0.35       # 35% Spot CVD Gewichtung  
futures_cvd_weight: float = 0.35    # 35% Futures CVD Gewichtung
smoothing_period: int = 5           # 5-Perioden Glättung
```

#### Long/Short Squeeze Berechnung (Zeile 165-185)
```python
# Long Squeeze Score Formel:
long_score = (
    price_factor * 0.3 +          # 30% Preis-Momentum
    divergence_factor * 0.4 +     # 40% CVD-Divergenz-Stärke  
    trend_factor * 0.3            # 30% CVD-Trend-Komponente
)

# Short Squeeze Score: Identische Gewichtung, invertierte Faktoren
short_score = -(price_factor * 0.3 + divergence_factor * 0.4 + trend_factor * 0.3)
```

#### Signal-Klassifikation (Zeile 233-244)
```python
def _classify_signal(self, score: float) -> str:
    if score <= -0.6:
        return "STRONG_LONG_SQUEEZE"    # Starkes Long Signal
    elif score <= -0.3:
        return "LONG_SQUEEZE"           # Schwaches Long Signal
    elif score >= 0.6:
        return "STRONG_SHORT_SQUEEZE"   # Starkes Short Signal  
    elif score >= 0.3:
        return "SHORT_SQUEEZE"          # Schwaches Short Signal
    else:
        return "NEUTRAL"                # Kein Signal
```

#### Lookback-Perioden (Zeile 47)
```python
self.lookback_periods = [5, 10, 15, 30, 60, 120, 240]  # Minuten
# 5,10,15,30min = Schnelle Reaktion
# 60,120,240min = Primäre Squeeze-Signale (1h,2h,4h)
```

#### Schwellenwerte (trading_parameters.yaml)
```yaml
squeeze_detection:
  confirmation_candles: 2              # Bestätigungs-Kerzen
  cvd_threshold: 1000000              # CVD-Schwellenwert (1M USD)
  price_momentum_threshold: 0.005     # 0.5% Preis-Momentum
  volume_threshold: 2.0               # 2x durchschnittliches Volumen
  min_score_threshold: 0.6            # Minimum Score für Entry
```

## 📊 EXCHANGE-KONFIGURATION (EXAKT)

### Aktivierte Exchanges (exchanges.yaml)
```yaml
binance:
  enabled: true
  rate_limit: 1200          # Requests/Minute
  testnet: true
  priority: 1               # Höchste Priorität

bybit:
  enabled: true
  rate_limit: 120
  testnet: true
  priority: 2

okx:
  enabled: true
  rate_limit: 60
  testnet: true
  priority: 3
```

### Exchange-Märkte (exchange_mapper.py)

#### BTC Markets (63 definierte Märkte - Verifiziert 2025)
```python
BTC_SPOT_MARKETS = [
    "BINANCE:btcusdt", "COINBASE:BTC-USD", "KRAKEN:XBTUSD",
    "BITFINEX:btcusd", "BYBIT:BTCUSDT", "OKX:BTC-USDT"
    # + 41 weitere Spot-Märkte (Total: 47 SPOT Markets)
]

BTC_PERP_MARKETS = [
    "BINANCE_FUTURES:btcusdt", "BYBIT:BTCUSDT", "OKX:BTC-USDT-SWAP",
    "DERIBIT:BTC-PERPETUAL", "BITMEX:XBTUSD"
    # + 11 weitere Perp-Märkte (Total: 16 PERP Markets)
]
```

#### ETH Markets (56 definierte Märkte - Verifiziert 2025)
```python
ETH_SPOT_MARKETS = [41 SPOT Märkte]  # BINANCE:ethusdt, COINBASE:ETH-USD, etc.
ETH_PERP_MARKETS = [15 PERP Märkte]  # BINANCE_FUTURES:ethusdt, BYBIT:ETHUSDT, etc.
```

#### Marktabdeckung-Verifikation (Juli 2025)
- **BTC Total**: 63 Märkte (47 SPOT + 16 PERP)
- **ETH Total**: 56 Märkte (41 SPOT + 15 PERP)  
- **Vollständige aggr-server Integration**: ✅ Alle Märkte korrekt klassifiziert
- **CVD-Datenqualität**: ✅ Reale Marktdaten mit -271M SPOT / -1.1B FUTURES CVD

## ⏰ TIMEFRAME-KONFIGURATION

### Haupt-Timeframes (trading_parameters.yaml)
```yaml
timeframes:
- 1m                        # Echtzeit-Signale
- 5m                        # Primäre Entry-Signale  
- 15m                       # Trend-Bestätigung
```

### Service-Timeframes (Erweitert)
```python
# squeezeflow_calculator.py - Lookback-Perioden
lookback_periods: [5, 10, 15, 30, 60, 120, 240]  # Minuten

# Verwendung:
# 5-15min: Schnelle Signal-Erkennung
# 30-60min: Entry-Timing-Optimierung  
# 120-240min: Trend-Bestätigung und Exit-Signale
```

## 🤖 MACHINE LEARNING INTEGRATION

### FreqAI Konfiguration (ml_config.yaml)
```yaml
freqai:
  enabled: true
  model_name: "LightGBMRegressorMultiTarget"
  train_period_days: 3              # 3 Tage Training
  backtest_period_days: 1           # 1 Tag Backtest
  live_retrain_hours: 6             # 6h Retrain-Intervall
  
  feature_parameters:
    include_timeframes: ["1m", "5m"]
    label_period_candles: 10
    indicator_periods_candles: [10, 20, 50]
    
  data_split_parameters:
    test_size: 0.33               # 33% Test-Daten
    shuffle: false                # Zeitreihen-Reihenfolge beibehalten
```

### Feature-Engineering
```python
# Technische Indikatoren
indicators = [
    'rsi', 'macd', 'bb_upperband', 'bb_lowerband', 
    'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26'
]

# Volume-basierte Features  
volume_features = [
    'volume_sma', 'volume_ema', 'vwap',
    'cvd_spot', 'cvd_perp', 'cvd_divergence'
]

# External Signals als Features
external_features = [
    'squeeze_score', 'signal_strength', 
    'price_momentum', 'volume_surge'
]
```

## 🛡️ RISIKOMANAGEMENT (EXAKTE WERTE)

### Position-Sizing (risk_management.yaml)
```yaml
position_sizing:
  max_position_size: 0.02        # 2% maximal pro Position
  max_total_exposure: 0.1        # 10% Gesamt-Exposure  
  min_position_size: 0.001       # 0.1% minimal pro Position
  
leverage:
  default: 1.0                   # Standard Leverage
  max_leverage: 3.0              # Maximum erlaubtes Leverage
```

### Risk Limits
```yaml
risk_limits:
  max_consecutive_losses: 5      # Max 5 Verluste hintereinander
  max_daily_loss: 0.05          # 5% maximaler Tagesverlust
  max_drawdown: 0.15            # 15% maximaler Drawdown
  daily_loss_reset_hour: 0      # Reset um Mitternacht UTC
```

### Stop Loss Konfiguration  
```yaml
stop_loss:
  default_pct: 0.02             # 2% Standard Stop Loss
  max_pct: 0.05                 # 5% maximaler Stop Loss
  trailing_enabled: true        # Trailing Stop aktiviert
  trailing_offset: 0.01         # 1% Trailing Offset
```

### Entry/Exit Schwellenwerte
```python
# Aus dem Code extrahierte Werte
ENTRY_THRESHOLDS = {
    'min_score': 0.6,            # Minimum Score für Position Entry
    'confirmation_score': 0.7,   # Score für Entry-Bestätigung
    'volume_multiplier': 2.0     # Mindest-Volume-Multiplikator
}

EXIT_THRESHOLDS = {
    'long_exit_score': 0.3,      # Long Exit bei Score > 0.3
    'short_exit_score': -0.3,    # Short Exit bei Score < -0.3
    'emergency_exit_score': 0.1, # Emergency Exit bei Score-Umkehr
    'profit_target': 0.04        # 4% Profit Target
}
```

## 🚀 TRADING STRATEGY (SqueezeFlowFreqAI)

### Strategy-Parameter (freqtrade/user_data/strategies/SqueezeFlowFreqAI.py)
```python
class SqueezeFlowFreqAI(IStrategy):
    # Grundkonfiguration
    timeframe = '5m'
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # ROI (Return on Investment) Tabelle
    minimal_roi = {
        "60": 0.01,    # Nach 60min: 1% ROI
        "30": 0.02,    # Nach 30min: 2% ROI  
        "0": 0.04      # Sofort: 4% ROI
    }
    
    # Stop Loss
    stoploss = -0.02   # 2% Stop Loss
    
    # Trailing Stop
    trailing_stop = True
    trailing_stop_positive = 0.01      # 1% positive trailing
    trailing_stop_positive_offset = 0.015  # 1.5% offset
```

### Signal-Integration
```python
# Redis-basierte externe Signal-Integration
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Hole Squeeze-Signal aus Redis
    signal_key = f"squeeze_signal:{metadata['pair']}:60"
    signal_data = self.redis_client.get(signal_key)
    
    # Entry-Bedingungen
    dataframe.loc[
        (
            (dataframe['squeeze_score'] <= -0.6) &        # Starkes Long Signal
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2) &  # 2x Volume
            (dataframe['rsi'] < 70) &                     # RSI nicht überkauft
            (dataframe['&-prediction'] > 0.5)             # ML Prediction positiv
        ),
        'enter_long'] = 1
    
    return dataframe
```

## 📈 MONITORING UND OBSERVABILITY

### Grafana Dashboards
```yaml
dashboards:
  - trading_performance:     # Echtzeit Trading Performance
      panels: [pnl_chart, position_status, trade_history]
      
  - squeeze_signals:         # Squeeze-Signal-Visualisierung  
      panels: [signal_heatmap, score_timeline, cvd_divergence]
      
  - system_health:          # System Health Monitoring
      panels: [service_status, resource_usage, error_rates]
      
  - ml_performance:         # ML-Modell Performance
      panels: [prediction_accuracy, feature_importance, model_metrics]
```

### Logging-Konfiguration
```python
logging_config = {
    'version': 1,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'data/logs/squeezeflow.log',
            'maxBytes': 100_000_000,    # 100MB pro Datei
            'backupCount': 5,           # 5 Backup-Dateien
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'squeezeflow': {'level': 'DEBUG', 'handlers': ['file']},
        'freqtrade': {'level': 'INFO', 'handlers': ['file']}
    }
}
```

## 🔧 ENTWICKLUNGSKOMMANDOS

### Hauptentry Point (main.py)
```bash
# System-Management
python main.py start                    # System starten
python main.py start --dry-run         # Dry-Run Modus (kein echtes Trading)
python main.py stop                    # System stoppen

# Backtesting & Optimierung  
python main.py backtest                # Standard Backtest
python main.py backtest --timerange 20240101-20240201
python main.py optimize               # Parameter-Optimierung

# Machine Learning
python main.py train-ml               # ML-Modell trainieren
python main.py train-ml --retrain     # Modell neu trainieren

# Testing & Validation
python main.py test                   # Vollständige System-Tests
python main.py test --component exchanges
python main.py download-data          # Historische Daten herunterladen
```

### Service-Management
```bash
# Docker Services
docker-compose up -d                  # Alle Services starten
docker-compose down                   # Alle Services stoppen
docker-compose logs -f [service]      # Service-Logs verfolgen

# Einzelne Services
docker-compose start aggr-server      # aggr-server starten
docker-compose restart freqtrade     # Freqtrade neustarten
```

### Initialisierung & Validierung
```bash
# Setup-Kommandos
python init.py --mode development     # Development Setup
python init.py --mode production      # Production Setup  
python init.py --force               # Force-Overwrite bestehender Configs

# Validierung & Status
python validate_setup.py             # Setup validieren
python status.py                     # System-Status prüfen
```

### Freqtrade-spezifische Kommandos
```bash
# Freqtrade CLI
freqtrade trade --config user_data/config.json --strategy SqueezeFlowFreqAI
freqtrade backtesting --config user_data/config.json --strategy SqueezeFlowFreqAI
freqtrade hyperopt --config user_data/config.json --strategy SqueezeFlowFreqAI
freqtrade plot-profit --config user_data/config.json
```

## 🔌 API-ENDPUNKTE

### Freqtrade REST API (Port 8080)
```python
base_url = "http://localhost:8080/api/v1"

endpoints = {
    # Trading Status
    'GET /status': 'Trading Status abrufen',
    'GET /profit': 'Profit-Informationen',
    'GET /trades': 'Aktuelle Trades',
    'GET /performance': 'Performance-Statistiken',
    
    # Trade Management  
    'POST /forceexit': 'Trade forciert beenden',
    'POST /forceentry': 'Trade forciert starten',
    'DELETE /trades/{trade_id}': 'Trade löschen',
    
    # System Control
    'POST /start': 'Trading starten',
    'POST /stop': 'Trading stoppen',
    'POST /reload_config': 'Konfiguration neu laden'
}
```

### aggr-server API (Port 3000)
```javascript
base_url = "http://localhost:3000"

endpoints = {
    // Echtzeit-Daten
    'GET /trades/:market': 'Aktuelle Trades für Market',
    'GET /ohlc/:market': 'OHLC-Daten für Market',
    'GET /liquidations/:market': 'Liquidations für Market',
    
    // Historische Daten
    'GET /historical/:market': 'Historische Trade-Daten',
    'GET /volume/:market': 'Volume-Daten',
    
    // WebSocket
    'WS /ws': 'Echtzeit WebSocket Stream'
}
```

### System-Endpunkte
```python
# Grafana (Port 3002)
'http://localhost:3002' - Monitoring Dashboards

# FreqUI (Port 8081)  
'http://localhost:8081' - Freqtrade Web Interface

# InfluxDB (Port 8086)
'http://localhost:8086' - InfluxDB Admin Interface

# Redis (Port 6379)
'redis://localhost:6379' - Redis Cache
```

## 🧪 TESTING UND VALIDIERUNG

### Test-Framework (main.py test)
```python
test_components = {
    'exchanges': 'Exchange-Konnektivität testen',
    'database': 'Datenbank-Verbindungen prüfen', 
    'redis': 'Redis-Cache-Funktionalität',
    'websockets': 'WebSocket-Verbindungen',
    'signals': 'Signal-Generierung validieren',
    'ml': 'ML-Modell-Funktionalität',
    'trading': 'Trading-Engine-Tests'
}

# Ausführung:
python main.py test                    # Alle Tests
python main.py test --component exchanges  # Spezifische Tests
```

### Backtesting-Engine (run_backtest.py)
```python
# Vordefinierte Zeiträume
backtest_periods = {
    'last_week': 'Letzte 7 Tage',
    'last_month': 'Letzter Monat',
    'january_2025': 'Januar 2025',
    'q1_2025': 'Q1 2025'
}

# Ausführung:
python run_backtest.py last_week         # Standard-Kapital
python run_backtest.py last_month 20000  # Mit 20k Start-Kapital
```

### Validierungs-Skripte
```python
# Setup-Validierung (validate_setup.py)
validation_checks = [
    'python_version',      # Python 3.8+ required
    'dependencies',        # Alle pip packages installiert
    'directories',         # Verzeichnisstruktur korrekt
    'configs',            # Alle Config-Dateien vorhanden
    'database',           # Datenbank-Verbindungen
    'docker',             # Docker Services
    'permissions'         # Dateiberechtigungen
]
```

## 🔐 SICHERHEIT UND CREDENTIALS

### API-Schlüssel-Management
```bash
# Umgebungsvariablen (.env)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_TESTNET=true

BYBIT_API_KEY=your_bybit_api_key  
BYBIT_SECRET_KEY=your_bybit_secret_key
BYBIT_TESTNET=true

OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key
OKX_PASSPHRASE=your_okx_passphrase
OKX_TESTNET=true
```

### Sicherheitsfeatures
```python
security_features = {
    'dry_run_mode': 'Simulation ohne echtes Geld',
    'testnet_support': 'Alle Exchanges mit Testnet',
    'api_key_encryption': 'Verschlüsselte Speicherung',
    'rate_limiting': 'API-Rate-Limiting eingebaut',
    'emergency_stop': 'Notfall-Stop bei kritischen Fehlern',
    'position_limits': 'Maximale Position-Größen',
    'drawdown_protection': 'Automatischer Stop bei hohem Drawdown'
}
```

## 📊 PERFORMANCE UND SKALIERUNG

### Resource-Konfiguration (docker-compose.yml)
```yaml
services:
  aggr-server:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          memory: 512M
          
  freqtrade:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
          
  influxdb:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### Performance-Optimierungen
```python
optimization_settings = {
    # InfluxDB
    'retention_policy': '30d',         # 30 Tage Datenaufbewahrung
    'batch_size': 10000,              # Batch-Schreibgröße
    'flush_interval': '10s',          # Flush-Intervall
    
    # Redis
    'max_memory': '256mb',            # Maximum Redis Memory
    'eviction_policy': 'allkeys-lru', # LRU Eviction
    
    # Python
    'asyncio_workers': 4,             # Async Worker-Anzahl
    'multiprocessing': True,          # Multiprocessing aktiviert
    'memory_limit': '4GB'             # Python Memory Limit
}
```

## 🔄 DEPLOYMENT UND LIFECYCLE

### Deployment-Varianten
```bash
# Vollständiges System-Deployment
./start.sh                           # Startet alle Services in korrekter Reihenfolge

# Entwicklung
python init.py --mode development    # Development-Setup
docker-compose -f docker-compose.dev.yml up

# Production  
python init.py --mode production     # Production-Setup
docker-compose up -d                # Detached Mode

# Custom
python init.py --force              # Force-Override bestehender Configs
```

### Lifecycle-Management
```bash
# Startup Sequence
1. InfluxDB + Redis (Dependencies)
2. aggr-server (Data Collection) 
3. squeezeflow-calculator (Signal Generation)
4. freqtrade (Trading Engine)
5. grafana + system-monitor (Monitoring)

# Shutdown Sequence  
1. freqtrade (Stop Trading)
2. squeezeflow-calculator (Stop Signals)
3. aggr-server (Stop Data Collection)
4. grafana + system-monitor
5. InfluxDB + Redis (Last)
```

### Health-Monitoring
```python
health_checks = {
    'docker_services': 'Alle 9 Container-Services',
    'database_connections': 'InfluxDB + Redis + SQLite',
    'api_endpoints': 'Freqtrade + aggr-server APIs',
    'websocket_connections': 'Exchange WebSocket Streams',
    'signal_generation': 'Squeeze-Signal-Pipeline',
    'ml_model_status': 'FreqAI-Modell-Status',
    'disk_space': 'Verfügbarer Speicherplatz',
    'memory_usage': 'Speicherverbrauch aller Services'
}

# Automatische Checks alle 60 Sekunden
python status.py --continuous --interval 60
```

## 📚 ABHÄNGIGKEITEN UND TECHNOLOGIE-STACK

### Core Python Dependencies
```txt
# Trading & Market Data
freqtrade>=2024.1                    # Trading Engine
ccxt>=4.0.0                         # Exchange Integration  
pandas>=2.0.0                       # Data Processing
numpy>=1.24.0                       # Numerical Computing
pandas-ta>=0.3.14b                  # Technical Analysis

# Databases & Caching
influxdb>=5.3.0                     # InfluxDB Client
redis>=4.5.0                        # Redis Client
SQLAlchemy>=2.0.0                   # SQL ORM

# Machine Learning
scikit-learn>=1.3.0                 # ML Framework
lightgbm>=4.0.0                     # Gradient Boosting
xgboost>=1.7.0                      # Alternative ML
joblib>=1.3.0                       # Model Persistence

# Async & Networking  
asyncio                             # Async Programming
aiohttp>=3.8.0                      # Async HTTP Client
websockets>=11.0.0                  # WebSocket Client
requests>=2.31.0                    # HTTP Requests

# Data Visualization & APIs
fastapi>=0.100.0                    # REST API Framework
uvicorn>=0.23.0                     # ASGI Server
plotly>=5.15.0                      # Interactive Plots
matplotlib>=3.7.0                   # Static Plots

# Configuration & Utilities
pyyaml>=6.0                         # YAML Processing
python-dotenv>=1.0.0                # Environment Variables
colorlog>=6.7.0                     # Colored Logging
tqdm>=4.65.0                        # Progress Bars
```

### Infrastructure Stack
```yaml
# Container Infrastructure  
docker: ">=20.10"                   # Container Runtime
docker-compose: ">=2.0"             # Multi-Container Apps

# Databases
influxdb: "1.8.10"                  # Time-Series Database
redis: "7-alpine"                   # In-Memory Cache
sqlite: ">=3.35"                    # Embedded Database

# Monitoring & Visualization
grafana: "latest"                   # Monitoring Dashboards
prometheus: "optional"              # Metrics Collection

# Data Collection
nodejs: ">=18"                      # aggr-server Runtime
npm: ">=8"                         # Node Package Manager
```

## 🚨 TROUBLESHOOTING UND DEBUG

### Häufige Probleme und Lösungen

#### 1. Docker-Services starten nicht
```bash
# Problem: Service-Start-Fehler
# Lösung:
docker-compose down --volumes       # Alle Services & Volumes stoppen
docker system prune -f             # Docker System aufräumen  
python init.py --force             # Configs neu erstellen
docker-compose up -d                # Services neu starten
```

#### 2. InfluxDB-Verbindungsfehler
```bash
# Problem: InfluxDB nicht erreichbar
# Debugging:
docker logs aggr-influx            # InfluxDB Logs prüfen
curl http://localhost:8086/ping    # Connectivity Test
python status.py --component database  # Database Status Check

# Lösung:
docker-compose restart aggr-influx
```

#### 3. Keine Squeeze-Signale generiert
```python
# Problem: Signal-Pipeline generiert keine Signale
# Debug-Steps:
1. python main.py test --component signals    # Signal-Tests
2. docker logs squeezeflow-calculator        # Service-Logs  
3. redis-cli KEYS "squeeze_signal:*"         # Cache prüfen
4. # InfluxDB Query für Datenvalidierung:
   SELECT * FROM squeeze_signals WHERE time > now() - 1h
```

#### 4. Freqtrade-Trading-Fehler
```bash
# Problem: Trading-Engine-Fehler
# Debug-Kommandos:
freqtrade show-trades --config user_data/config.json
freqtrade test-pairlist --config user_data/config.json  
docker logs freqtrade | grep ERROR

# Häufige Fixes:
freqtrade download-data --config user_data/config.json --timerange 20240101-
```

### Debug-Logging aktivieren
```python
# Ausführliches Debug-Logging
export SQUEEZEFLOW_DEBUG=true
export FREQTRADE_LOG_LEVEL=DEBUG

# Log-Dateien überwachen
tail -f data/logs/squeezeflow.log
tail -f user_data/logs/freqtrade.log
```

### Performance-Debugging
```bash
# System-Resource-Monitoring
docker stats                       # Container-Resource-Usage
python status.py --performance     # Performance-Metriken
htop                              # System-Ressourcen

# Datenbank-Performance
influx -execute 'SHOW DIAGNOSTICS'
redis-cli INFO memory
```

## 🔧 GRAFANA DASHBOARD ENTWICKLUNG - KRITISCHE ERKENNTNISSE

### InfluxDB 1.8 + Grafana 2025 Kompatibilitätsprobleme

**WICHTIGES PROBLEM (2025):** Moderne Grafana-Versionen haben Breaking Changes in der InfluxDB-Datasource-Konfiguration eingeführt.

#### Veraltete Konfiguration (funktioniert NICHT mehr):
```yaml
# ❌ VERALTET - Verursacht HTTP 400 Fehler
datasources:
  - name: InfluxDB
    type: influxdb
    database: significant_trades    # ← Veraltet!
    httpMode: GET                   # ← Suboptimal
```

#### Korrekte moderne Konfiguration:
```yaml
# ✅ MODERN - Funktioniert mit Grafana 2025
datasources:
  - name: InfluxDB
    type: influxdb
    jsonData:
      dbName: significant_trades    # ← Neues Format!
      httpMode: POST               # ← Empfohlen für InfluxDB 1.8
      timeInterval: "5s"
    secureJsonData:
      password: ""
```

#### Dashboard-Query-Probleme und Lösungen:

**PROBLEM:** Raw SQL-Queries in Dashboards verursachen HTTP 400 Fehler in modernen Grafana-Versionen.

**❌ Fehlerhafte Raw SQL-Syntax:**
```json
{
  "targets": [{
    "query": "SELECT raw_score FROM squeeze_signals WHERE symbol = 'BTCUSDT' AND time > now() - 6h",
    "refId": "A"
  }]
}
```

**✅ Korrekte Query Builder-Syntax:**
```json
{
  "targets": [{
    "datasource": {
      "type": "influxdb",
      "uid": "P951FEA4DE68E13C5"
    },
    "measurement": "squeeze_signals",
    "select": [
      [{"type": "field", "params": ["raw_score"]}, {"type": "mean", "params": []}]
    ],
    "tags": [
      {"key": "symbol", "operator": "=", "value": "BTCUSDT"}
    ],
    "groupBy": [
      {"type": "time", "params": ["$__interval"]},
      {"type": "fill", "params": ["null"]}
    ],
    "refId": "A"
  }]
}
```

#### Retention Policy-Referenzen:

**❌ Falsche Escaping-Syntax:**
```sql
FROM "aggr_1m"."trades_1m"    # Verursacht JSON-Parse-Fehler
```

**✅ Korrekte JSON-Escaping-Syntax:**
```sql
FROM \"aggr_1m\".\"trades_1m\"  # Korrekt escaped für JSON
```

#### Debug-Strategie für Dashboard-Probleme:

1. **Grafana-Logs prüfen:**
```bash
docker logs squeezeflow-grafana --tail 50 | grep "status=400"
```

2. **Direkte InfluxDB-Queries testen:**
```bash
docker exec squeezeflow-grafana wget -qO- "http://aggr-influx:8086/query?db=significant_trades&q=SELECT+*+FROM+squeeze_signals+LIMIT+5"
```

3. **JSON-Syntax validieren:**
```bash
python3 -m json.tool dashboard.json > /dev/null && echo "OK" || echo "BROKEN"
```

### Lösungsschritte für Dashboard-Reparatur:

1. **Datasource modernisieren** - `database` → `jsonData.dbName`
2. **HTTP-Modus auf POST** - bessere InfluxDB 1.8 Kompatibilität  
3. **Raw SQL zu Query Builder konvertieren** - moderne Grafana-Anforderung
4. **JSON-Escaping korrigieren** - Retention Policy-Namen richtig escapen
5. **Dashboard-UIDs vereinheitlichen** - korrekte Datasource-Referenzen

### Command-Referenz für Grafana-Debugging:
```bash
# Dashboard JSON-Syntax prüfen
find config/grafana/dashboards -name "*.json" -exec python3 -m json.tool {} \; > /dev/null 2>&1

# Grafana-Container neu starten mit sauberen Dashboards  
docker restart squeezeflow-grafana

# InfluxDB-Konnektivität aus Grafana-Container testen
docker exec squeezeflow-grafana ping -c 3 aggr-influx
```

**MERKE:** Diese Breaking Changes treten nur bei Grafana 2025+ in Kombination mit InfluxDB 1.8 auf. Legacy-Setups funktionieren weiterhin mit älteren Grafana-Versionen.

## 📋 ERWEITERTE KONFIGURATIONSSZENARIEN

### Multi-Exchange-Setup
```yaml
# config/exchanges.yaml - Erweiterte Konfiguration
exchanges:
  binance:
    enabled: true
    markets: ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    rate_limit: 1200
    testnet: true
    priority: 1
    
  bybit:
    enabled: true  
    markets: ["BTCUSDT", "ETHUSDT"]
    rate_limit: 120
    testnet: true
    priority: 2
    
  okx:
    enabled: false                  # Temporär deaktiviert
    markets: ["BTC-USDT", "ETH-USDT"]
    rate_limit: 60
    testnet: true
    priority: 3
```

### Erweiterte Risk Management
```yaml
# config/risk_management.yaml - Erweiterte Regeln
risk_profiles:
  conservative:
    max_position_size: 0.01        # 1% pro Position
    max_total_exposure: 0.05       # 5% Gesamt-Exposure
    max_daily_loss: 0.02          # 2% Daily Loss Limit
    
  aggressive:
    max_position_size: 0.05        # 5% pro Position  
    max_total_exposure: 0.20       # 20% Gesamt-Exposure
    max_daily_loss: 0.10          # 10% Daily Loss Limit

# Zeitbasierte Regeln
time_based_rules:
  trading_hours:
    enabled: true
    start_hour: 6                  # UTC
    end_hour: 22                   # UTC
    
  weekend_trading:
    enabled: false                 # Kein Weekend-Trading
```

### ML-Modell-Konfiguration erweitert
```yaml
# config/ml_config.yaml - Erweiterte ML-Settings
freqai:
  models:
    primary:
      model_name: "LightGBMRegressorMultiTarget"
      train_period_days: 3
      live_retrain_hours: 6
      
    secondary:
      model_name: "XGBoostRegressor"  
      train_period_days: 7
      live_retrain_hours: 12
      
  ensemble:
    enabled: true
    voting_method: "soft"          # Soft Voting für Predictions
    weights: [0.7, 0.3]           # Primary: 70%, Secondary: 30%
    
  feature_engineering:
    technical_indicators: true
    volume_profile: true
    order_flow: true
    external_signals: true
    correlation_features: true
```

## 📖 FAZIT UND SYSTEMREIFE

SqueezeFlow Trader 2 ist ein **hochentwickeltes, produktionsreifes Trading-System** mit folgenden Schlüsselstärken:

### ✅ Technische Exzellenz
- **Vollständige Containerisierung** für nahtloses Deployment
- **Multi-Exchange-Integration** mit 20+ Exchanges  
- **Innovative Squeeze-Detection** basierend auf CVD-Divergenz
- **ML-Integration** mit FreqAI und LightGBM
- **Robuste Mikroservice-Architektur**
- **Umfassendes Monitoring** mit Grafana
- **Professionelles Risikomanagement**

### 🎯 Trading-Kompetenz  
- **Wissenschaftlich fundierte Strategie** mit exakten Parametern
- **Multi-Timeframe-Analyse** (1m, 5m, 15m + erweiterte Lookbacks)
- **Echtzeit-Signal-Generierung** mit Redis-Caching
- **Backtesting-Framework** für Strategievalidierung
- **Dry-Run-Modus** für sicheres Testing

### 🔧 Entwickler-Freundlichkeit
- **Clean Code-Architektur** mit klarer Modularität
- **Umfassende Konfigurierbarkeit** aller Parameter
- **Automatisierte Setup-Validierung** und Health-Checks
- **Ausführliche Dokumentation** in Code und Configs
- **Debugging-Support** mit detailliertem Logging

**Das System stellt eine professionelle, institutionelle Trading-Lösung dar, die höchste Ansprüche an Qualität, Sicherheit und Performance erfüllt.**

---

*Diese Dokumentation wird automatisch bei Systemänderungen aktualisiert. Letzte Aktualisierung: $(date)*