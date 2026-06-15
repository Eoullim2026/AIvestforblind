import asyncio
import json
import websockets
from datetime import datetime

WS_URL = "ws://220.69.21.83:8765"


async def main():
    event = {
        "name": "자전거",
        "direction": "우측 접근",
        "distance": "0.8m",
        "confidence": "96%",
        "level": "HIGH",
        "message": "우측 자전거 접근 위험",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps(event, ensure_ascii=False))
        print("sent:", event)


if __name__ == "__main__":
    asyncio.run(main())
