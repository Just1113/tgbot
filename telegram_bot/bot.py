from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ContextTypes, filters
)
import logging
from datetime import datetime
import json
import asyncio
from typing import Optional

from app.config import TradingConfig
from core.trading_engine import TradingEngine

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, config: TradingConfig, trading_engine: TradingEngine):
        self.config = config
        self.engine = trading_engine
        
        # Initialize bot application
        self.application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        
        # Register handlers
        self._register_handlers()
        
        # Admin users
        self.admin_users = [config.TELEGRAM_CHAT_ID]
    
    def _register_handlers(self):
        """Register Telegram command handlers"""
        # Command handlers
        handlers = [
            CommandHandler("start", self.start),
            CommandHandler("help", self.help),
            CommandHandler("status", self.status),
            CommandHandler("trades", self.show_trades),
            CommandHandler("balance", self.show_balance),
            CommandHandler("metrics", self.show_metrics),
            CommandHandler("positions", self.show_positions),
            CommandHandler("start_bot", self.start_bot),
            CommandHandler("stop_bot", self.stop_bot),
            CommandHandler("close_all", self.close_all),
            CommandHandler("set_leverage", self.set_leverage),
            CommandHandler("set_risk", self.set_risk),
            CommandHandler("config", self.show_config),
            CommandHandler("logs", self.show_logs, filters=filters.User(user_id=int(self.config.TELEGRAM_CHAT_ID)) if self.config.TELEGRAM_CHAT_ID else None),
        ]
        
        for handler in handlers:
            self.application.add_handler(handler)
        
        # Callback query handler
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send welcome message"""
        welcome_text = f"""
🤖 *High Frequency Scalping Bot*
        
*Status:* {'✅ Running' if self.engine.is_running else '❌ Stopped'}
*Symbol:* {self.config.SYMBOL}
*Leverage:* {self.config.LEVERAGE}x
        
*Available Commands:*
/status - Bot status
/trades - Recent trades
/balance - Account balance
/metrics - Performance metrics
/positions - Open positions
        
*Admin Commands:*
/start_bot - Start trading
/stop_bot - Stop trading
/close_all - Close all positions
/set_leverage <1-100> - Change leverage
/set_risk <sl%> <tp%> <max_loss%> - Adjust risk
/config - Show current configuration
        """
        
        keyboard = [
            [InlineKeyboardButton("🔄 Status", callback_data="status"),
             InlineKeyboardButton("📊 Metrics", callback_data="metrics")],
            [InlineKeyboardButton("💰 Balance", callback_data="balance"),
             InlineKeyboardButton("📈 Positions", callback_data="positions")],
            [InlineKeyboardButton("▶️ Start Bot", callback_data="start_bot"),
             InlineKeyboardButton("⏹️ Stop Bot", callback_data="stop_bot")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_text, 
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message"""
        help_text = """
*🤖 Bot Help Guide*
        
*High-Frequency Scalping Bot Features:*
• Machine Learning based predictions
• Real-time market analysis
• Automated risk management
• Telegram notifications
• Comprehensive trade logging
        
*Risk Parameters:*
• Stop Loss: 0.3-2.0%
• Take Profit: 0.2-1.5%
• Max Daily Loss: 1-5%
        
*Important Notes:*
1. Always start with testnet
2. Monitor performance regularly
3. Adjust parameters based on market conditions
4. Never risk more than you can afford to lose
        
Use /status to check current bot state.
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot status"""
        try:
            # Get account balance
            balance = self.engine.client.get_account_balance()
            usdt_balance = float(balance.get('totalEquity', 0)) if balance else 0
            
            # Get current price
            market_data = self.engine.client.get_market_data()
            current_price = float(market_data.get('lastPrice', 0)) if market_data else 0
            
            status_text = f"""
*🤖 Bot Status*
        
• *Running:* {'✅ Yes' if self.engine.is_running else '❌ No'}
• *Symbol:* {self.config.SYMBOL}
• *Leverage:* {self.config.LEVERAGE}x
• *Balance:* ${usdt_balance:,.2f}
• *Current Price:* ${current_price:,.2f}
        
• *Open Trades:* {len(self.engine.open_trades)}/{self.config.MAX_OPEN_TRADES}
• *Daily P&L:* ${self.engine.metrics.get('daily_pnl', 0):.2f}
• *Total Trades:* {self.engine.metrics.get('total_trades', 0)}
• *Win Rate:* {self.engine.metrics.get('win_rate', 0):.1f}%
        
*Risk Settings:*
• Stop Loss: {self.config.STOP_LOSS_PCT}%
• Take Profit: {self.config.TAKE_PROFIT_PCT}%
• Max Position: {self.config.MAX_POSITION_SIZE}
• Max Daily Loss: {self.config.MAX_DAILY_LOSS_PCT}%
            """
            
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_status"),
                 InlineKeyboardButton("📊 Metrics", callback_data="show_metrics")],
                [InlineKeyboardButton("📈 Start Bot", callback_data="start_bot"),
                 InlineKeyboardButton("🛑 Stop Bot", callback_data="stop_bot")],
                [InlineKeyboardButton("💰 Balance", callback_data="balance"),
                 InlineKeyboardButton("📊 Trades", callback_data="trades")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                status_text, 
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text(f"Error getting status: {str(e)}")
    
    async def show_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show recent trades"""
        try:
            # Get recent trades from engine (you'll need to implement this)
            trades_text = "*📊 Recent Trades*\n\n"
            
            if not self.engine.open_trades and self.engine.metrics['total_trades'] == 0:
                trades_text += "No trades yet."
            else:
                # Show open trades
                if self.engine.open_trades:
                    trades_text += "*Open Trades:*\n"
                    for trade_id, trade in list(self.engine.open_trades.items())[:5]:
                        trade_time = trade.opened_at.strftime("%H:%M:%S")
                        trades_text += f"""
• *{trade.side}* {trade.quantity:.4f} {trade.symbol}
  Entry: ${trade.entry_price:.2f}
  SL: ${trade.stop_loss:.2f} | TP: ${trade.take_profit:.2f}
  Time: {trade_time}
                        """
                
                # Show recent closed trades
                trades_text += "\n*Recent Performance:*\n"
                trades_text += f"""
• Total Trades: {self.engine.metrics['total_trades']}
• Winning: {self.engine.metrics['winning_trades']}
• Losing: {self.engine.metrics['losing_trades']}
• Win Rate: {self.engine.metrics.get('win_rate', 0):.1f}%
• Total P&L: ${self.engine.metrics['total_pnl']:.2f}
                """
            
            await update.message.reply_text(trades_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in trades command: {e}")
            await update.message.reply_text(f"Error getting trades: {str(e)}")
    
    async def show_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show account balance"""
        try:
            balance = self.engine.client.get_account_balance()
            
            if not balance:
                await update.message.reply_text("Could not fetch balance.")
                return
            
            usdt_balance = float(balance.get('totalEquity', 0))
            available = float(balance.get('availableToWithdraw', 0))
            unrealized_pnl = float(balance.get('totalUnrealisedPnl', 0))
            
            balance_text = f"""
*💰 Account Balance*
        
• *Total Equity:* ${usdt_balance:,.2f}
• *Available:* ${available:,.2f}
• *Unrealized P&L:* ${unrealized_pnl:,.2f}
• *Margin Used:* ${usdt_balance - available:,.2f}
            """
            
            # Get open positions
            positions = self.engine.client.get_open_positions()
            if positions:
                balance_text += "\n*Open Positions:*"
                for pos in positions:
                    size = float(pos.get('size', 0))
                    if size > 0:
                        symbol = pos.get('symbol', '')
                        side = pos.get('side', '')
                        entry = float(pos.get('avgPrice', 0))
                        current = float(pos.get('markPrice', 0))
                        pnl = float(pos.get('unrealisedPnl', 0))
                        pnl_pct = float(pos.get('unrealisedPnlPcnt', 0)) * 100
                        
                        balance_text += f"""
• {symbol}: {side} {size}
  Entry: ${entry:.2f} | Current: ${current:.2f}
  P&L: ${pnl:.2f} ({pnl_pct:.2f}%)
                        """
            
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="balance")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                balance_text, 
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error in balance command: {e}")
            await update.message.reply_text(f"Error getting balance: {str(e)}")
    
    async def show_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show performance metrics"""
        try:
            metrics = self.engine.metrics
            
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            
            metrics_text = f"""
*📈 Performance Metrics*
        
• *Total Trades:* {metrics.get('total_trades', 0)}
• *Winning Trades:* {metrics.get('winning_trades', 0)}
• *Losing Trades:* {metrics.get('losing_trades', 0)}
• *Win Rate:* {win_rate:.1f}%
• *Profit Factor:* {profit_factor:.2f}
        
• *Total P&L:* ${metrics.get('total_pnl', 0):.2f}
• *Daily P&L:* ${metrics.get('daily_pnl', 0):.2f}
• *Max Drawdown:* {metrics.get('max_drawdown', 0):.2f}%
        
• *Open Trades:* {len(self.engine.open_trades)}
• *Bot Uptime:* {metrics.get('uptime', 'N/A')}
            """
            
            keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="metrics")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                metrics_text, 
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error in metrics command: {e}")
            await update.message.reply_text(f"Error getting metrics: {str(e)}")
    
    async def show_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show open positions"""
        try:
            positions = self.engine.client.get_open_positions()
            
            if not positions or all(float(p.get('size', 0)) == 0 for p in positions):
                await update.message.reply_text("No open positions.")
                return
            
            positions_text = "*📊 Open Positions*\n\n"
            
            for pos in positions:
                size = float(pos.get('size', 0))
                if size > 0:
                    symbol = pos.get('symbol', '')
                    side = pos.get('side', '')
                    entry = float(pos.get('avgPrice', 0))
                    current = float(pos.get('markPrice', 0))
                    pnl = float(pos.get('unrealisedPnl', 0))
                    pnl_pct = float(pos.get('unrealisedPnlPcnt', 0)) * 100
                    leverage = float(pos.get('leverage', 0))
                    
                    positions_text += f"""
• *{symbol}*
  Side: {side}
  Size: {size:.4f}
  Entry: ${entry:.2f}
  Current: ${current:.2f}
  P&L: ${pnl:.2f} ({pnl_pct:.2f}%)
  Leverage: {leverage}x
                    """
            
            keyboard = [[
                InlineKeyboardButton("❌ Close All", callback_data="close_all_positions"),
                InlineKeyboardButton("🔄 Refresh", callback_data="positions")
            ]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                positions_text, 
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error in positions command: {e}")
            await update.message.reply_text(f"Error getting positions: {str(e)}")
    
    async def show_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current configuration"""
        config_text = f"""
*⚙️ Current Configuration*
        
*API Settings:*
• Testnet: {'✅ Yes' if self.config.BYBIT_TESTNET else '❌ No (LIVE)'}
• Symbol: {self.config.SYMBOL}
        
*Trading Parameters:*
• Leverage: {self.config.LEVERAGE}x
• Max Position Size: {self.config.MAX_POSITION_SIZE}
• Max Open Trades: {self.config.MAX_OPEN_TRADES}
        
*Risk Management:*
• Stop Loss: {self.config.STOP_LOSS_PCT}%
• Take Profit: {self.config.TAKE_PROFIT_PCT}%
• Max Daily Loss: {self.config.MAX_DAILY_LOSS_PCT}%
• Auto Leverage: {'✅ On' if self.config.ENABLE_AUTO_LEVERAGE_ADJUSTMENT else '❌ Off'}
        
*ML Settings:*
• Prediction Confidence: {self.config.PREDICTION_CONFIDENCE*100:.1f}%
• Volatility Threshold: {self.config.VOLATILITY_THRESHOLD}
        
*Performance:*
• Data Interval: {self.config.DATA_FETCH_INTERVAL}s
• Order Timeout: {self.config.ORDER_TIMEOUT}s
        """
        
        await update.message.reply_text(config_text, parse_mode='Markdown')
    
    async def show_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show recent logs (admin only)"""
        try:
            # Read last 20 lines from log file
            log_file = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-20:]
                    logs = ''.join(lines)
            except:
                logs = "No logs found or unable to read log file."
            
            log_text = f"""
*📋 Recent Logs*
        
```{logs}```
            """
            
            await update.message.reply_text(log_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in logs command: {e}")
            await update.message.reply_text(f"Error getting logs: {str(e)}")
    
    async def start_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start the trading bot"""
        if str(update.effective_user.id) not in self.admin_users:
            await update.message.reply_text("❌ Admin only command.")
            return
        
        if not self.engine.is_running:
            await self.engine.start()
            await update.message.reply_text("✅ Bot started successfully.")
        else:
            await update.message.reply_text("⚠️ Bot is already running.")
    
    async def stop_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop the trading bot"""
        if str(update.effective_user.id) not in self.admin_users:
            await update.message.reply_text("❌ Admin only command.")
            return
        
        if self.engine.is_running:
            await self.engine.stop()
            await update.message.reply_text("🛑 Bot stopped successfully.")
        else:
            await update.message.reply_text("⚠️ Bot is already stopped.")
    
    async def close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close all positions"""
        if str(update.effective_user.id) not in self.admin_users:
            await update.message.reply_text("❌ Admin only command.")
            return
        
        await update.message.reply_text("🔄 Closing all positions...")
        await self.engine.close_all_positions()
        await update.message.reply_text("✅ All positions closed.")
    
    async def set_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set leverage"""
        if str(update.effective_user.id) not in self.admin_users:
            await update.message.reply_text("❌ Admin only command.")
            return
        
        if context.args and context.args[0].isdigit():
            leverage = int(context.args[0])
            if 1 <= leverage <= 100:
                self.config.LEVERAGE = leverage
                success = self.engine.client._set_leverage(leverage)
                if success:
                    await update.message.reply_text(f"✅ Leverage set to {leverage}x")
                else:
                    await update.message.reply_text("❌ Failed to set leverage on exchange")
            else:
                await update.message.reply_text("❌ Leverage must be between 1-100")
        else:
            await update.message.reply_text("Usage: /set_leverage <1-100>")
    
    async def set_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set risk parameters"""
        if str(update.effective_user.id) not in self.admin_users:
            await update.message.reply_text("❌ Admin only command.")
            return
        
        if len(context.args) == 3:
            try:
                sl = float(context.args[0])
                tp = float(context.args[1])
                max_loss = float(context.args[2])
                
                # Validate ranges
                if 0.1 <= sl <= 5.0 and 0.1 <= tp <= 5.0 and 0.5 <= max_loss <= 10.0:
                    self.config.STOP_LOSS_PCT = sl
                    self.config.TAKE_PROFIT_PCT = tp
                    self.config.MAX_DAILY_LOSS_PCT = max_loss
                    
                    await update.message.reply_text(
                        f"✅ Risk parameters updated:\n"
                        f"• Stop Loss: {sl}%\n"
                        f"• Take Profit: {tp}%\n"
                        f"• Max Daily Loss: {max_loss}%"
                    )
                else:
                    await update.message.reply_text(
                        "❌ Invalid values. Use:\n"
                        "• Stop Loss: 0.1-5.0%\n"
                        "• Take Profit: 0.1-5.0%\n"
                        "• Max Daily Loss: 0.5-10.0%"
                    )
            except ValueError:
                await update.message.reply_text("❌ Invalid values. Use numbers.")
        else:
            await update.message.reply_text(
                "Usage: /set_risk <stop_loss%> <take_profit%> <max_daily_loss%>\n"
                "Example: /set_risk 0.5 0.3 2.0"
            )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        callback_data = query.data
        
        if callback_data == "refresh_status":
            await self.status(update, context)
        elif callback_data == "show_metrics":
            await self.show_metrics(update, context)
        elif callback_data == "start_bot":
            await self.start_bot(update, context)
        elif callback_data == "stop_bot":
            await self.stop_bot(update, context)
        elif callback_data == "close_all_positions":
            await self.close_all(update, context)
        elif callback_data == "balance":
            await self.show_balance(update, context)
        elif callback_data == "positions":
            await self.show_positions(update, context)
        elif callback_data == "trades":
            await self.show_trades(update, context)
        elif callback_data == "metrics":
            await self.show_metrics(update, context)
        elif callback_data == "status":
            await self.status(update, context)
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Telegram bot error: {context.error}")
        
        try:
            # Notify admin of error
            error_msg = f"❌ *Bot Error*\n\nError: {str(context.error)[:200]}"
            await self.send_notification(error_msg)
        except:
            pass
    
    async def send_notification(self, message: str):
        """Send notification to admin chat"""
        try:
            await self.application.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def run_polling(self):
        """Run the Telegram bot with polling"""
        logger.info("Starting Telegram bot polling...")
        
        try:
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                close_loop=False
            )
        except Exception as e:
            logger.error(f"Telegram bot failed: {e}")
