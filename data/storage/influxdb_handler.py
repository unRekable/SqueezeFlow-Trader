"""
InfluxDB handler for trading system data
Manages all trading-related measurements in InfluxDB
"""

from influxdb import InfluxDBClient
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json


class InfluxDBHandler:
    """Handles all InfluxDB operations for the trading system"""
    
    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        self.client = InfluxDBClient(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database
        )
        self.logger = logging.getLogger(__name__)
        
    def write_squeeze_signal(self, 
                           symbol: str,
                           exchange: str, 
                           signal_type: str,
                           squeeze_score: float,
                           price_change: float = None,
                           volume_surge: float = None,
                           oi_change: float = None,
                           cvd_divergence: float = None,
                           timestamp: datetime = None):
        """Write squeeze signal to InfluxDB"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        point = {
            "measurement": "squeeze_signals",
            "tags": {
                "symbol": symbol,
                "exchange": exchange,
                "signal_type": signal_type
            },
            "time": timestamp,
            "fields": {
                "squeeze_score": float(squeeze_score)
            }
        }
        
        # Add optional fields
        if price_change is not None:
            point["fields"]["price_change"] = float(price_change)
        if volume_surge is not None:
            point["fields"]["volume_surge"] = float(volume_surge)
        if oi_change is not None:
            point["fields"]["oi_change"] = float(oi_change)
        if cvd_divergence is not None:
            point["fields"]["cvd_divergence"] = float(cvd_divergence)
            
        try:
            self.client.write_points([point], time_precision='s')
            self.logger.debug(f"Wrote squeeze signal for {symbol}: {squeeze_score}")
        except Exception as e:
            self.logger.error(f"Failed to write squeeze signal: {e}")
    
    def write_position(self,
                      position_id: str,
                      symbol: str,
                      exchange: str,
                      side: str,
                      entry_price: float,
                      size: float,
                      status: str = "open",
                      exit_price: float = None,
                      stop_loss: float = None,
                      take_profit: float = None,
                      pnl: float = None,
                      pnl_percentage: float = None,
                      fees: float = 0.0,
                      is_dry_run: bool = True,
                      strategy_name: str = "SqueezeFlow",
                      timestamp: datetime = None):
        """Write trading position to InfluxDB"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        point = {
            "measurement": "positions",
            "tags": {
                "symbol": symbol,
                "exchange": exchange,
                "side": side,
                "status": status,
                "strategy_name": strategy_name,
                "is_dry_run": str(is_dry_run).lower()
            },
            "time": timestamp,
            "fields": {
                "position_id": position_id,
                "entry_price": float(entry_price),
                "size": float(size),
                "fees": float(fees)
            }
        }
        
        # Add optional fields
        if exit_price is not None:
            point["fields"]["exit_price"] = float(exit_price)
        if stop_loss is not None:
            point["fields"]["stop_loss"] = float(stop_loss)
        if take_profit is not None:
            point["fields"]["take_profit"] = float(take_profit)
        if pnl is not None:
            point["fields"]["pnl"] = float(pnl)
        if pnl_percentage is not None:
            point["fields"]["pnl_percentage"] = float(pnl_percentage)
            
        try:
            self.client.write_points([point], time_precision='s')
            self.logger.debug(f"Wrote position {position_id} for {symbol}")
        except Exception as e:
            self.logger.error(f"Failed to write position: {e}")
    
    def write_trade(self,
                   trade_id: str,
                   position_id: str,
                   symbol: str,
                   exchange: str,
                   side: str,
                   price: float,
                   amount: float,
                   fee: float = 0.0,
                   order_type: str = "market",
                   order_id: str = None,
                   is_dry_run: bool = True,
                   timestamp: datetime = None):
        """Write trade execution to InfluxDB"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        point = {
            "measurement": "trades",
            "tags": {
                "symbol": symbol,
                "exchange": exchange,
                "side": side,
                "order_type": order_type,
                "is_dry_run": str(is_dry_run).lower()
            },
            "time": timestamp,
            "fields": {
                "trade_id": trade_id,
                "position_id": position_id,
                "price": float(price),
                "amount": float(amount),
                "fee": float(fee)
            }
        }
        
        if order_id:
            point["fields"]["order_id"] = order_id
            
        try:
            self.client.write_points([point], time_precision='s')
            self.logger.debug(f"Wrote trade {trade_id} for {symbol}")
        except Exception as e:
            self.logger.error(f"Failed to write trade: {e}")
    
    def write_ml_prediction(self,
                          symbol: str,
                          model_name: str,
                          prediction: float,
                          confidence: float = None,
                          feature_importance: Dict = None,
                          timestamp: datetime = None):
        """Write ML prediction to InfluxDB"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        point = {
            "measurement": "ml_predictions",
            "tags": {
                "symbol": symbol,
                "model_name": model_name
            },
            "time": timestamp,
            "fields": {
                "prediction": float(prediction)
            }
        }
        
        if confidence is not None:
            point["fields"]["confidence"] = float(confidence)
        if feature_importance is not None:
            point["fields"]["feature_importance"] = json.dumps(feature_importance)
            
        try:
            self.client.write_points([point], time_precision='s')
            self.logger.debug(f"Wrote ML prediction for {symbol}: {prediction}")
        except Exception as e:
            self.logger.error(f"Failed to write ML prediction: {e}")
    
    def write_system_metric(self,
                          metric_name: str,
                          metric_value: float,
                          component: str,
                          metric_unit: str = None,
                          timestamp: datetime = None):
        """Write system metric to InfluxDB"""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        point = {
            "measurement": "system_metrics",
            "tags": {
                "metric_name": metric_name,
                "component": component
            },
            "time": timestamp,
            "fields": {
                "metric_value": float(metric_value)
            }
        }
        
        if metric_unit:
            point["tags"]["metric_unit"] = metric_unit
            
        try:
            self.client.write_points([point], time_precision='s')
            self.logger.debug(f"Wrote system metric {metric_name}: {metric_value}")
        except Exception as e:
            self.logger.error(f"Failed to write system metric: {e}")
    
    def get_latest_squeeze_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get latest squeeze signals from InfluxDB"""
        
        where_clause = ""
        if symbol:
            where_clause = f"WHERE symbol = '{symbol}'"
            
        query = f"""
            SELECT * FROM squeeze_signals 
            {where_clause}
            ORDER BY time DESC 
            LIMIT {limit}
        """
        
        try:
            result = self.client.query(query)
            return list(result.get_points())
        except Exception as e:
            self.logger.error(f"Failed to get squeeze signals: {e}")
            return []
    
    def get_open_positions(self, symbol: str = None) -> List[Dict]:
        """Get open trading positions"""
        
        where_clause = "WHERE status = 'open'"
        if symbol:
            where_clause += f" AND symbol = '{symbol}'"
            
        query = f"""
            SELECT * FROM positions 
            {where_clause}
            ORDER BY time DESC
        """
        
        try:
            result = self.client.query(query)
            return list(result.get_points())
        except Exception as e:
            self.logger.error(f"Failed to get open positions: {e}")
            return []
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio performance summary"""
        
        queries = {
            "total_positions": "SELECT COUNT(*) FROM positions",
            "open_positions": "SELECT COUNT(*) FROM positions WHERE status = 'open'", 
            "closed_positions": "SELECT COUNT(*) FROM positions WHERE status = 'closed'",
            "total_pnl": "SELECT SUM(pnl) FROM positions WHERE status = 'closed'",
            "win_rate": """
                SELECT 
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins,
                    COUNT(*) as total
                FROM positions 
                WHERE status = 'closed'
            """
        }
        
        summary = {}
        for key, query in queries.items():
            try:
                result = self.client.query(query)
                points = list(result.get_points())
                if points:
                    if key == "win_rate":
                        wins = points[0].get('wins', 0)
                        total = points[0].get('total', 0)
                        summary[key] = (wins / total * 100) if total > 0 else 0
                    else:
                        summary[key] = list(points[0].values())[0] if points[0] else 0
                else:
                    summary[key] = 0
            except Exception as e:
                self.logger.error(f"Failed to get {key}: {e}")
                summary[key] = 0
        
        return summary
    
    def test_connection(self) -> bool:
        """Test InfluxDB connection"""
        try:
            self.client.ping()
            return True
        except Exception as e:
            self.logger.error(f"InfluxDB connection test failed: {e}")
            return False
    
    def write_trading_data(self, measurement: str, points: List[Dict]) -> bool:
        """Write trading data points to InfluxDB"""
        try:
            influx_points = []
            for point in points:
                influx_point = {
                    "measurement": measurement,
                    "tags": point.get('tags', {}),
                    "time": point.get('timestamp'),
                    "fields": point.get('fields', {})
                }
                influx_points.append(influx_point)
            
            self.client.write_points(influx_points, time_precision='s')
            self.logger.debug(f"Wrote {len(influx_points)} points to {measurement}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write trading data to {measurement}: {e}")
            return False
    
    def close(self):
        """Close InfluxDB connection"""
        if self.client:
            self.client.close()