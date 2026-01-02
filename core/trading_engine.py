import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time

from app.config import TradingConfig
from core.bybit_client import BybitClient
from core.ml_predictor import MLPredictor
from core.risk_manager import RiskManager
from app.database import log_trade, update_trade, get_bot_setting, set_bot_setting

logger = logging.getLogger(__name__)

class TradeStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

@dataclass
class Trade:
    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    status: TradeStatus
    opened_at: datetime
    order_id: Optional[str] = None
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert trade to dictionary for database"""
        data = asdict(self)
        data['status'] = self.status.value
        data['metadata'] = {
            'version': '2.0',
            'strategy': 'ml_scalping'
        }
        return data

class TradingEngine:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.client = BybitClient(config)
        self.ml_predictor = MLPredictor(config.ML_MODEL_PATH)
        self.risk_manager = RiskManager(config)
        
        # State management
        self.open_trades: Dict[str, Trade] = {}
        self.is_running = False
        self.last_ml_training = None
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        
        # Market data cache
        self.market_data_cache = {
            'last_update': 0,
            'data': None,
            'cache_duration': 1  # seconds
        }
        
        # Load previous state
        asyncio.create_task(self._load_state())
    
    async def _load_state(self):
        """Load bot state from database"""
        try:
            # Load metrics
            saved_metrics = await get_bot_setting('metrics')
            if saved_metrics:
                self.metrics.update(saved_metrics)
                self.logger.info("Loaded metrics from database")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    async def _save_state(self):
        """Save bot state to database"""
        try:
            self.metrics['last_updated'] = datetime.now().isoformat()
            await set_bot_setting('metrics', self.metrics)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    async def start(self):
        """Start the trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine is already running")
            return
        
        self.is_running = True
        self.logger.info("Trading engine started")
        
        # Initialize ML model
        await self._initialize_ml_model()
        
        # Start main trading loop
        asyncio.create_task(self._trading_loop())
        
        # Start maintenance tasks
        asyncio.create_task(self._maintenance_loop())
    
    async def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        
        # Save state
        await self._save_state()
        
        # Close all open positions
        await self.close_all_positions()
        
        self.logger.info("Trading engine stopped")
    
    async def _initialize_ml_model(self):
        """Initialize or train ML model"""
        try:
            # Check if model exists and is recent
            model_age_hours = 24
            train_needed = True
            
            # Try to load existing model
            if self.ml_predictor.load_model(self.config.ML_MODEL_PATH):
                self.logger.info("Loaded existing ML model")
                train_needed = False
            
            # Fetch historical data for training if needed
            if train_needed:
                historical_data = await self._fetch_historical_data(limit=5000)
                
                if historical_data is not None and len(historical_data) > 100:
                    self.logger.info("Training ML model with historical data...")
                    self.ml_predictor.train_model(historical_data)
                    self.last_ml_training = datetime.now()
                else:
                    self.logger.warning("Insufficient data for ML training")
            else:
                self.last_ml_training = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ML model: {e}")
    
    async def _trading_loop(self):
        """Main trading loop optimized for Render"""
        self.logger.info("Starting trading loop")
        
        while self.is_running:
            try:
                cycle_start = time.time()
                
                # Fetch current market data
                market_data = await self._get_market_data()
                
                if market_data is None or market_data.empty:
                    await asyncio.sleep(1)
                    continue
                
                # Get ML prediction
                prediction, confidence = self.ml_predictor.predict(market_data)
                
                # Check risk management rules
                if not self.risk_manager.can_trade():
                    await asyncio.sleep(1)
                    continue
                
                # Generate trading signal
                signal = self._generate_signal(prediction, confidence, market_data)
                
                # Execute trade if conditions are met
                if self._should_execute_trade(signal):
                    await self._execute_trade(signal)
                
                # Monitor open positions
                await self._monitor_positions()
                
                # Update metrics periodically
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    await self._update_metrics()
                    await self._save_state()
                
                # Calculate sleep time to maintain interval
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.config.DATA_FETCH_INTERVAL - cycle_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait longer on error
    
    def _should_execute_trade(self, signal: Dict) -> bool:
        """Determine if trade should be executed"""
        # Check signal strength
        if signal['strength'] < self.config.PREDICTION_CONFIDENCE:
            return False
        
        # Check max open trades
        if len(self.open_trades) >= self.config.MAX_OPEN_TRADES:
            return False
        
        # Check if we already have a position in same direction
        for trade in self.open_trades.values():
            if trade.side == signal['side']:
                return False
        
        # Additional filters can be added here
        return True
    
    async def _execute_trade(self, signal: Dict):
        """Execute a trade based on signal"""
        try:
            # Place the order with stop loss and take profit
            order_result = self.client.place_order(
                side=signal['side'],
                quantity=signal['quantity'],
                order_type="Market",
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
            
            if order_result:
                # Create trade record
                trade = Trade(
                    id=order_result['orderId'],
                    symbol=self.config.SYMBOL,
                    side=signal['side'],
                    quantity=signal['quantity'],
                    entry_price=signal['price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    leverage=self.config.LEVERAGE,
                    status=TradeStatus.OPEN,
                    opened_at=datetime.now(),
                    order_id=order_result['orderId']
                )
                
                # Add to open trades
                self.open_trades[trade.id] = trade
                
                # Log the trade to database
                await log_trade(trade.to_dict())
                
                # Update metrics
                self.metrics['total_trades'] += 1
                
                self.logger.info(
                    f"Trade executed: {trade.side} {trade.quantity} {trade.symbol} "
                    f"at ${trade.entry_price:.2f}"
                )
                
                # Update risk manager
                self.risk_manager.trade_opened()
                
        except Exception as e:
            self.logger.error(f"Failed to execute trade: {e}", exc_info=True)
    
    async def _monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            # Get current positions from exchange
            positions = self.client.get_open_positions()
            
            for position in positions:
                size = float(position['size'])
                if size == 0:
                    continue
                
                position_id = position['positionId']
                
                # Check if we have this trade in our records
                trade = self.open_trades.get(position_id)
                
                if trade:
                    # Update trade with current position data
                    current_price = float(position['markPrice'])
                    unrealized_pnl = float(position['unrealisedPnl'])
                    
                    # Check exit conditions
                    await self._check_exit_conditions(trade, current_price, unrealized_pnl)
                
        except Exception as e:
            self.logger.error(f"Failed to monitor positions: {e}")
    
    async def _check_exit_conditions(self, trade: Trade, current_price: float, unrealized_pnl: float):
        """Check if trade should be exited"""
        exit_reason = None
        
        if trade.side == "Buy":
            # Long position
            if current_price <= trade.stop_loss:
                exit_reason = "stop_loss"
            elif current_price >= trade.take_profit:
                exit_reason = "take_profit"
        else:
            # Short position
            if current_price >= trade.stop_loss:
                exit_reason = "stop_loss"
            elif current_price <= trade.take_profit:
                exit_reason = "take_profit"
        
        # Additional exit conditions
        if not exit_reason:
            # Check time-based exit (e.g., don't hold overnight)
            hold_time = datetime.now() - trade.opened_at
            if hold_time.total_seconds() > 3600:  # 1 hour max
                exit_reason = "time_exit"
        
        if exit_reason:
            await self._close_trade(trade, current_price, exit_reason)
    
    async def _close_trade(self, trade: Trade, exit_price: float, reason: str):
        """Close a trade"""
        try:
            # Place closing order
            close_side = "Sell" if trade.side == "Buy" else "Buy"
            
            # Use reduce-only order to close position
            order_result = self.client.place_order(
                side=close_side,
                quantity=trade.quantity,
                order_type="Market",
                reduce_only=True
            )
            
            if order_result:
                # Update trade record
                trade.closed_at = datetime.now()
                trade.exit_price = exit_price
                trade.status = TradeStatus.CLOSED
                trade.reason = reason
                
                # Calculate P&L
                if trade.side == "Buy":
                    trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                else:
                    trade.pnl = (trade.entry_price - exit_price) * trade.quantity
                
                trade.pnl_percentage = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
                
                # Update metrics
                self.metrics['total_pnl'] += trade.pnl
                self.metrics['daily_pnl'] += trade.pnl
                
                if trade.pnl > 0:
                    self.metrics['winning_trades'] += 1
                else:
                    self.metrics['losing_trades'] += 1
                
                # Calculate win rate
                if self.metrics['total_trades'] > 0:
                    self.metrics['win_rate'] = (
                        self.metrics['winning_trades'] / self.metrics['total_trades']
                    ) * 100
                
                # Update risk manager
                self.risk_manager.trade_closed(trade.pnl)
                
                # Update trade in database
                await update_trade(trade.id, trade.to_dict())
                
                # Remove from open trades
                self.open_trades.pop(trade.id, None)
                
                self.logger.info(
                    f"Trade closed: {trade.id} "
                    f"P&L: ${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%) "
                    f"Reason: {reason}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to close trade: {e}", exc_info=True)
    
    async def close_all_positions(self):
        """Close all open positions"""
        self.logger.info("Closing all positions...")
        
        for trade_id, trade in list(self.open_trades.items()):
            try:
                market_data = self.client.get_market_data(trade.symbol)
                if market_data:
                    current_price = float(market_data.get('lastPrice', trade.entry_price))
                    await self._close_trade(trade, current_price, "manual_close")
            except Exception as e:
                self.logger.error(f"Failed to close trade {trade_id}: {e}")
        
        # Also close any positions not in our records
        positions = self.client.get_open_positions()
        for position in positions:
            size = float(position['size'])
            if size > 0:
                self.client.close_position()
    
    async def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Get current market data with caching"""
        current_time = time.time()
        
        # Return cached data if still valid
        if (self.market_data_cache['data'] is not None and 
            current_time - self.market_data_cache['last_update'] < self.market_data_cache['cache_duration']):
            return self.market_data_cache['data']
        
        try:
            # Get ticker data
            ticker = self.client.get_market_data(self.config.SYMBOL)
            
            # Get kline data for indicators
            klines = self.client.get_kline_data(interval=self.config.TIMEFRAME, limit=100)
            
            if ticker and klines:
                # Process klines into DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                # Set index
                df.set_index('timestamp', inplace=True)
                
                # Update cache
                self.market_data_cache['data'] = df
                self.market_data_cache['last_update'] = current_time
                
                return df
            
        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
        
        return None
    
    def _generate_signal(self, prediction: float, confidence: float, 
                        market_data: pd.DataFrame) -> Dict:
        """Generate trading signal based on ML prediction"""
        current_price = market_data['close'].iloc[-1]
        
        # Calculate signal strength
        signal_strength = prediction * confidence
        
        # Determine side
        if prediction > 0.5:
            side = "Buy"
            stop_loss = current_price * (1 - self.config.STOP_LOSS_PCT / 100)
            take_profit = current_price * (1 + self.config.TAKE_PROFIT_PCT / 100)
        else:
            side = "Sell"
            stop_loss = current_price * (1 + self.config.STOP_LOSS_PCT / 100)
            take_profit = current_price * (1 - self.config.TAKE_PROFIT_PCT / 100)
        
        # Calculate position size based on risk
        position_size = self.risk_manager.calculate_position_size(
            current_price, stop_loss
        )
        
        # Apply maximum position size
        position_size = min(position_size, self.config.MAX_POSITION_SIZE)
        
        return {
            'side': side,
            'strength': signal_strength,
            'prediction': prediction,
            'confidence': confidence,
            'price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'quantity': position_size,
            'timestamp': datetime.now()
        }
    
    async def _fetch_historical_data(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch historical price data"""
        try:
            klines = self.client.get_kline_data(interval=self.config.TIMEFRAME, limit=limit)
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                df.set_index('timestamp', inplace=True)
                return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {e}")
        
        return None
    
    async def _update_metrics(self):
        """Update performance metrics"""
        # Calculate additional metrics
        if self.metrics['winning_trades'] > 0 and self.metrics['losing_trades'] > 0:
            avg_win = self.metrics['total_pnl'] / self.metrics['winning_trades']
            avg_loss = abs(self.metrics['total_pnl'] / self.metrics['losing_trades'])
            self.metrics['profit_factor'] = avg_win / avg_loss if avg_loss > 0 else 0
        
        self.metrics['last_updated'] = datetime.now().isoformat()
    
    async def _maintenance_loop(self):
        """Maintenance tasks loop"""
        while self.is_running:
            try:
                # Retrain ML model periodically
                if self.last_ml_training:
                    hours_since_training = (datetime.now() - self.last_ml_training).total_seconds() / 3600
                    if hours_since_training >= self.config.ML_TRAIN_INTERVAL_HOURS:
                        self.logger.info("Retraining ML model...")
                        historical_data = await self._fetch_historical_data(limit=5000)
                        if historical_data is not None:
                            self.ml_predictor.train_model(historical_data)
                            self.last_ml_training = datetime.now()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _cleanup_old_data(self):
        """Clean up old data from database"""
        # Implement if needed
        pass
    
    def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            'running': self.is_running,
            'open_trades': len(self.open_trades),
            'metrics': self.metrics,
            'config': {
                'symbol': self.config.SYMBOL,
                'leverage': self.config.LEVERAGE,
                'position_size': self.config.MAX_POSITION_SIZE
            }
        }
