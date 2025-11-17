import os
from typing import Optional
import asyncio
from .IBKR_trading import TradingApp 

HOST = os.getenv("IB_HOST")
PORT = int(os.getenv("IB_PORT"))
# CLIENT_ID = int(os.getenv("IB_CLIENT_ID"))

def generate_client_id():
    # fallback if no env is set
    import os, time
    return (os.getpid() % 10000) + (int(time.time()) % 1000)


class IBKRSession:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None
    ):
        # Resolve values (env → fallback → provided)
        self.host = host or HOST
        self.port = port or PORT
        self.client_id = client_id or generate_client_id()

        # TradingApp *requires* host/port/client_id → always pass them
        self.ib = TradingApp()

    async def __aenter__(self):
        await self._connect()
        return self.ib  # return the TradingApp instance

    async def __aexit__(self, exc_type, exc, tb):
        await self._disconnect()

    async def _connect(self):
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
        except AttributeError:
            await self.ib.start(self.host, self.port, clientId=self.client_id)

    async def _disconnect(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.ib.stop)