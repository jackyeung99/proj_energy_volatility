# pip install ib-insync pandas

import os
import asyncio
import logging
from typing import List, Optional

import pandas as pd
from ib_insync import IB, Stock, Order, util

# hush chatty logs so host/IP banners don't show
logging.getLogger("ib_insync").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)


class TradingApp:
    def __init__(self) -> None:
        self.ib = IB()

    # ---------- lifecycle ----------
    async def start(self, host: str, port: int, clientId: int) -> None:
        # connect async; ib_insync runs the reader thread for you
        await self.ib.connectAsync(host, port, clientId=clientId)

        if self.ib.isConnected():
            print("Connected to Interactive Brokers successfully.")
        else:
            print("Failed to connect to Interactive Brokers.")


    async def stop(self) -> None:
        if self.ib.isConnected():
            await self.ib.disconnectAsync()

    # ---------- helpers ----------
    async def _qualified_stock(self, symbol: str) -> Optional[Stock]:
        c = Stock(symbol, "SMART", "USD")
        try:
            (qc,) = await self.ib.qualifyContractsAsync(c)
            return qc
        except Exception:
            return None

    # ---------- data ----------
    async def fetch_one_symbol(
        self,
        symbol: str,
        duration: str = "2 Y",
        bar_size: str = "1 day",
        what: str = "TRADES",
        rth: bool = True,
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Daily bars for one US stock.
        Returns: DataFrame[date, Symbol, open, high, low, close, volume] or None
        """
        qc = await self._qualified_stock(symbol)
        if qc is None:
            return None

        for attempt in range(max_retries):
            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    qc,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what,
                    useRTH=rth,
                    formatDate=1,
                    keepUpToDate=False,
                )
                if not bars:
                    return None
                df = util.df(bars)
                df = df.rename(columns=str.lower)
                df["symbol"] = symbol
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                df = df[["date", "symbol", "open", "high", "low", "close", "volume"]].dropna()
                # Align with your original column casing if you prefer:
                df = df.rename(columns={"symbol": "Symbol"})
                return df
            except Exception:
                if attempt + 1 == max_retries:
                    return None
                # backoff to respect pacing (354) & transient hiccups
                await asyncio.sleep(1 + 2 * attempt)

        return None

    async def fetch_many_symbols(
        self,
        symbols: List[str],
        duration: str = "2 Y",
        bar_size: str = "1 day",
        what: str = "TRADES",
        rth: bool = True,
        concurrency: int = 5,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Concurrency-limited fetch across many symbols.
        Returns: long DataFrame[date, Symbol, open, high, low, close, volume]
        """
        sem = asyncio.Semaphore(concurrency)

        async def _guarded(sym: str):
            async with sem:
                return await self.fetch_one_symbol(
                    sym, duration, bar_size, what, rth, max_retries
                )

        results = await asyncio.gather(*(_guarded(s) for s in symbols))
        dfs = [df for df in results if df is not None and not df.empty]
        if not dfs:
            return pd.DataFrame(columns=["date", "Symbol", "open", "high", "low", "close", "volume"])
        return pd.concat(dfs, ignore_index=True)

    # ---------- orders ----------
    async def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MKT",
        **kwargs,
    ) -> Optional[int]:
        """
        Places a basic order (default Market). Returns orderId (int) or None on failure.
        """
        qc = await self._qualified_stock(symbol)
        if qc is None:
            return None

        o = Order(
            action=action,
            totalQuantity=quantity,
            orderType=order_type,
            **kwargs,
        )
        trade = self.ib.placeOrder(qc, o)
        # wait until the order is acknowledged or filled
        try:
            await trade.orderStatusEvent  # lightweight await
        except Exception:
            pass
        return trade.order.orderId

