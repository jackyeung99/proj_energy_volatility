from typing import Dict, Optional
import pandas as pd
from ibapi.client import *
from ibapi.wrapper import *
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import BarData
import numpy as np

class TradingApp(EClient, EWrapper):

    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {}
        self.nextOrderId: Optional[int] = None
        self.end_idx = []

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderReject='') -> None:
        print(f"Error: {reqId}, {errorCode}, {errorString}")

    def nextValidId(self, orderId: int) -> None:
        super().nextValidId(orderId)
        self.nextOrderId = orderId

    def parse_bar_date(self,bar_date: str) -> pd.Timestamp:
        """
        Parse IBKR bar.date into pandas Timestamp.
        Handles intraday ('YYYYMMDD HH:MM:SS') and daily ('YYYYMMDD') formats.
        """
        if len(bar_date) == 8:  # daily bars
            return pd.to_datetime(bar_date, format="%Y%m%d")
        else:  # intraday bars
            return pd.to_datetime(bar_date, format="%Y%m%d %H:%M:%S %Z")

    def get_historical_data(self, reqId: int, contract: Contract, endDateTime,durationStr,barSizeSetting,whatToShow,useRTH,formatDate,keepUpToDate) -> pd.DataFrame:

        self.data[reqId] = pd.DataFrame(columns=["time", "high", "low", "close","volume"])
        self.data[reqId].set_index("time", inplace=True)
        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            endDateTime=endDateTime,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            useRTH=useRTH,
            formatDate=formatDate,
            keepUpToDate=keepUpToDate,
            chartOptions=[],
        )
        return self.data
    

    def historicalData(self, reqId: int, bar: BarData) -> None:
        print("HistoricalData. ReqId:", reqId, "date.", bar.date)
        # date = pd.to_datetime(datetime.datetime.fromtimestamp(int(bar.date)).strftime('%Y-%m-%d %H:%M:%S'))
        date = self.parse_bar_date(bar.date)
        self.data[reqId].loc[
            date,
            ["open", "high", "low", "close","volume"]
        ] = [bar.open, bar.high, bar.low, bar.close, bar.volume]

        self.data[reqId].loc[date,["open", "high", "low", "close"]] = self.data[reqId].loc[date,["open", "high", "low", "close"]].astype(float)
        self.data[reqId].loc[date,"volume"] = int(self.data[reqId].loc[date,"volume"])
        

    def historicalDataUpdate(self, reqId: int, bar: BarData) -> None:
        print("HistoricalDataUpdate. ReqId:", reqId, "date.", bar.date)
        # date = pd.to_datetime(datetime.datetime.fromtimestamp(int(bar.date)).strftime('%Y-%m-%d %H:%M:%S'))
        date = self.parse_bar_date(bar.date)
        self.data[reqId].loc[
            date,
            ["open", "high", "low", "close","volume"]
        ] = [bar.open, bar.high, bar.low, bar.close, bar.volume]

        self.data[reqId].loc[date,["open", "high", "low", "close"]] = self.data[reqId].loc[date,["open", "high", "low", "close"]].astype(float)
        self.data[reqId].loc[date,"volume"] = int(self.data[reqId].loc[date,"volume"])

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)
        # self.data = self.data_update
        self.end_idx.append(reqId)
        print("Finished")

    @staticmethod
    def get_contract(symbol: str) -> Contract:

        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def place_order(self, contract: Contract, action: str, order_type: str, quantity: int) -> None:

        order = Order()
        order.action = action
        order.orderType = order_type
        order.totalQuantity = quantity

        self.placeOrder(self.nextOrderId, contract, order)
        self.nextOrderId += 1

    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        self.execution = execution
        print("Order placed")
        return self.execution
    

