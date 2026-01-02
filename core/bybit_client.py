from pybit.unified_trading import HTTP
import aiohttp
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import logging
from app.config import TradingConfig

logger = logging.getLogger(__name__)

class BybitClient:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger
        
        # Initialize HTTP client with retry configuration
        self.session = HTTP(
            testnet=config.BYBIT_TESTNET,
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
            recv_window=5000  # 5 seconds
        )
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Initialize leverage
        self._initialize_account()
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _initialize_account(self):
        """Initialize account settings"""
        try:
            # Set leverage
            self._set_leverage(self.config.LEVERAGE)
            
            # Set margin mode to isolated
            self._set_margin_mode()
            
            self.logger.info("Account initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize account: {e}")
    
    def _set_leverage(self, leverage: int):
        """Set leverage for the trading pair"""
        self._rate_limit()
        try:
            response = self.session.set_leverage(
                category="linear",
                symbol=self.config.SYMBOL,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            if response['retCode'] == 0:
                self.logger.info(f"Leverage set to {leverage}x for {self.config.SYMBOL}")
                return True
            else:
                self.logger.error(f"Failed to set leverage: {response['retMsg']}")
                return False
        except Exception as e:
            self.logger.error(f"Exception setting leverage: {e}")
            return False
    
    def _set_margin_mode(self):
        """Set margin mode to isolated"""
        self._rate_limit()
        try:
            response = self.session.switch_margin_mode(
                category="linear",
                symbol=self.config.SYMBOL,
                tradeMode=1,  # 1 for isolated margin
                buyLeverage=str(self.config.LEVERAGE),
                sellLeverage=str(self.config.LEVERAGE)
            )
            return response['retCode'] == 0
        except Exception as e:
            self.logger.error(f"Failed to set margin mode: {e}")
            return False
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        self._rate_limit()
        try:
            response = self.session.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
            if response['retCode'] == 0:
                return response['result']['list'][0]
            else:
                self.logger.error(f"Failed to get balance: {response['retMsg']}")
                return {}
        except Exception as e:
            self.logger.error(f"Exception getting balance: {e}")
            return {}
    
    def get_market_data(self, symbol: str = None) -> Dict:
        """Get market data for symbol"""
        self._rate_limit()
        try:
            symbol = symbol or self.config.SYMBOL
            response = self.session.get_tickers(
                category="linear",
                symbol=symbol
            )
            if response['retCode'] == 0 and response['result']['list']:
                return response['result']['list'][0]
            else:
                self.logger.error(f"Failed to get market data: {response.get('retMsg', 'Unknown error')}")
                return {}
        except Exception as e:
            self.logger.error(f"Exception getting market data: {e}")
            return {}
    
    def get_kline_data(self, interval: str = "1", limit: int = 200) -> List[Dict]:
        """Get kline/candlestick data"""
        self._rate_limit()
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=self.config.SYMBOL,
                interval=interval,
                limit=limit
            )
            if response['retCode'] == 0:
                return response['result']['list']
            else:
                self.logger.error(f"Failed to get kline data: {response['retMsg']}")
                return []
        except Exception as e:
            self.logger.error(f"Exception getting kline data: {e}")
            return []
    
    def place_order(self, 
                    side: str, 
                    quantity: float, 
                    order_type: str = "Market",
                    price: float = None,
                    reduce_only: bool = False,
                    stop_loss: float = None,
                    take_profit: float = None) -> Optional[Dict]:
        """Place an order with optional stop loss and take profit"""
        self._rate_limit()
        try:
            order_params = {
                "category": "linear",
                "symbol": self.config.SYMBOL,
                "side": side,
                "orderType": order_type,
                "qty": str(quantity),
                "timeInForce": "GTC",
                "reduceOnly": reduce_only
            }
            
            if price and order_type == "Limit":
                order_params["price"] = str(price)
            
            if stop_loss:
                order_params["stopLoss"] = str(stop_loss)
            
            if take_profit:
                order_params["takeProfit"] = str(take_profit)
            
            response = self.session.place_order(**order_params)
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                self.logger.info(f"Order placed: {order_id} - {side} {quantity} {self.config.SYMBOL}")
                return response['result']
            else:
                self.logger.error(f"Order failed: {response['retMsg']}")
                return None
                
        except Exception as e:
            self.logger.error(f"Exception placing order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order"""
        self._rate_limit()
        try:
            response = self.session.cancel_order(
                category="linear",
                symbol=self.config.SYMBOL,
                orderId=order_id
            )
            success = response['retCode'] == 0
            if success:
                self.logger.info(f"Order cancelled: {order_id}")
            else:
                self.logger.error(f"Failed to cancel order: {response['retMsg']}")
            return success
        except Exception as e:
            self.logger.error(f"Exception cancelling order: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        self._rate_limit()
        try:
            response = self.session.cancel_all_orders(
                category="linear",
                symbol=self.config.SYMBOL
            )
            success = response['retCode'] == 0
            if success:
                self.logger.info("All orders cancelled")
            else:
                self.logger.error(f"Failed to cancel orders: {response['retMsg']}")
            return success
        except Exception as e:
            self.logger.error(f"Exception cancelling all orders: {e}")
            return False
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        self._rate_limit()
        try:
            response = self.session.get_positions(
                category="linear",
                symbol=self.config.SYMBOL
            )
            if response['retCode'] == 0:
                return response['result']['list']
            else:
                self.logger.error(f"Failed to get positions: {response['retMsg']}")
                return []
        except Exception as e:
            self.logger.error(f"Exception getting positions: {e}")
            return []
    
    def get_order_history(self, limit: int = 100) -> List[Dict]:
        """Get order history"""
        self._rate_limit()
        try:
            response = self.session.get_order_history(
                category="linear",
                symbol=self.config.SYMBOL,
                limit=limit
            )
            if response['retCode'] == 0:
                return response['result']['list']
            else:
                self.logger.error(f"Failed to get order history: {response['retMsg']}")
                return []
        except Exception as e:
            self.logger.error(f"Exception getting order history: {e}")
            return []
    
    def get_funding_rate(self) -> Optional[float]:
        """Get current funding rate"""
        self._rate_limit()
        try:
            response = self.session.get_funding_rate_history(
                category="linear",
                symbol=self.config.SYMBOL,
                limit=1
            )
            if response['retCode'] == 0 and response['result']['list']:
                return float(response['result']['list'][0]['fundingRate'])
            return None
        except Exception as e:
            self.logger.error(f"Exception getting funding rate: {e}")
            return None
    
    def close_position(self, side: str = None) -> bool:
        """Close current position"""
        self._rate_limit()
        try:
            params = {
                "category": "linear",
                "symbol": self.config.SYMBOL,
                "positionIdx": 0  # One-way mode
            }
            
            if side:
                params["side"] = side
            
            response = self.session.set_trading_stop(**params)
            return response['retCode'] == 0
        except Exception as e:
            self.logger.error(f"Exception closing position: {e}")
            return False
    
    async def get_realtime_data(self) -> Dict:
        """Get real-time data via REST (fallback for Render)"""
        try:
            data = self.get_market_data()
            kline = self.get_kline_data(limit=1)
            
            return {
                'price': float(data.get('lastPrice', 0)),
                'bid': float(data.get('bid1Price', 0)),
                'ask': float(data.get('ask1Price', 0)),
                'volume': float(data.get('volume24h', 0)),
                'high': float(data.get('highPrice24h', 0)),
                'low': float(data.get('lowPrice24h', 0)),
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"Failed to get realtime data: {e}")
            return {}
