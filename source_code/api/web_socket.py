import asyncio
import threading
import uuid
from typing import Dict

import websockets

server = None


class WebSocketServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.connected_clients: Dict[str] = {}
        self.server = None

    async def register_client(self, websocket):
        """注册新客户端连接"""
        client_id = str(uuid.uuid4())
        # client_id = websocket.request.path.split("/")[-1]
        # try:
        #     _ = uuid.UUID(client_id)
        # except Exception as e:
        #     print(f"Invalid client_id: {client_id}")
        #     raise e
        self.connected_clients[client_id] = websocket
        print(f"New client connected: {client_id}. Total clients: {len(self.connected_clients)}")
        return client_id

    async def unregister_client(self, client_id: str):
        """移除断开连接的客户端"""
        if client_id in self.connected_clients:
            del self.connected_clients[client_id]
            print(f"Client disconnected: {client_id}. Total clients: {len(self.connected_clients)}")

    async def handler(self, websocket):
        """处理客户端连接"""
        client_id = await self.register_client(websocket)
        try:
            async for message in websocket:
                # 这里可以处理客户端发来的消息
                print(f"Received message from {client_id}: {message}")
                # 可以回复消息
                # await websocket.send(f"Server received: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(client_id)

    async def broadcast(self, message: str):
        if not self.connected_clients:
            print("No clients connected to broadcast to.")
            return

        disconnected_clients = []
        coros = []
        for client_id, websocket in self.connected_clients.items():
            coros.append(self._safe_send(websocket, client_id, message, disconnected_clients))
        await asyncio.gather(*coros)

        for client_id in disconnected_clients:
            await self.unregister_client(client_id)

    async def _safe_send(self, websocket, client_id, message, disconnected_clients):
        try:
            await websocket.send(message)
            print(f"Message sent to {client_id}")
        except websockets.exceptions.ConnectionClosed:
            disconnected_clients.append(client_id)

    async def send_message(self, client_id: str, message: str):
        """向指定客户端发送消息"""
        if client_id not in self.connected_clients:
            print(f"Client {client_id} not found.")
            return
        try:
            await self.connected_clients[client_id].send(message)
            print(f"Message sent to {client_id}")
        except websockets.exceptions.ConnectionClosed:
            print(f"Client {client_id} disconnected.")
            await self.unregister_client(client_id)

    async def start(self):
        """启动WebSocket服务器"""
        self.server = await websockets.serve(self.handler, self.host, self.port)
        print(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("WebSocket server stopped")
        self.connected_clients.clear()


# 使用示例
async def main():
    global server
    try:
        server = WebSocketServer()
        await server.start()
        while True:
            await asyncio.sleep(30)
            print("websocket server is running")
    except KeyboardInterrupt:
        await server.stop()


def start_async_in_thread():
    asyncio.run(main())


thread = threading.Thread(target=start_async_in_thread, name="AsyncThread")
thread.start()
