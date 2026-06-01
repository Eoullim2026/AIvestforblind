import asyncio
import json
from datetime import datetime

import websockets


HOST = "0.0.0.0"
PORT = 8765

clients = set()


def now_time():
    return datetime.now().strftime("%H:%M:%S")


def normalize_event(data):
    return {
        "name": data.get("name", data.get("object", "unknown")),
        "direction": data.get("direction", "방향 미확인"),
        "distance": data.get("distance", "-"),
        "confidence": data.get("confidence", "-"),
        "level": str(data.get("level", "INFO")).upper(),
        "message": data.get("message", ""),
        "vibration": bool(data.get("vibration", False)),
        "timestamp": data.get("timestamp", now_time()),
    }


async def broadcast(message):
    if not clients:
        return

    payload = json.dumps(message, ensure_ascii=False)
    dead_clients = []

    for client in clients:
        try:
            await client.send(payload)
        except Exception:
            dead_clients.append(client)

    for client in dead_clients:
        clients.discard(client)


async def handler(websocket):
    clients.add(websocket)
    print(f"[{now_time()}] client connected. total={len(clients)}")

    try:
        async for raw_message in websocket:
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                print(f"[{now_time()}] invalid json: {raw_message}")
                continue

            event = normalize_event(data)

            print(f"[{now_time()}] event received: {event}")

            await broadcast(event)

    except websockets.ConnectionClosed:
        pass

    finally:
        clients.discard(websocket)
        print(f"[{now_time()}] client disconnected. total={len(clients)}")


async def main():
    async with websockets.serve(handler, HOST, PORT):
        print(f"event websocket relay server running on ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
