"""
Batch Stress Testing Script for Cognitive CanSat WebSocket Serial Bridge
Connects to ws://127.0.0.1:8000/ws/serial and streams 500 rapid telemetry packets to evaluate bridge throughput and latency.
"""

import asyncio
import websockets
import time
import random

WS_URL = "ws://127.0.0.1:8000/ws/serial"

async def stress_test():
    try:
        async with websockets.connect(WS_URL) as websocket:
            print(f"Successfully connected to {WS_URL}")
            print("Sending 500 rapid telemetry packets...")

            start_time = time.time()

            for _ in range(500):
                temp = round(random.uniform(20.0, 35.0), 2)
                pressure = round(random.uniform(900.0, 1013.25), 2)
                altitude = round(random.uniform(0.0, 500.0), 2)

                # Simulated raw CanSat CSV packet
                telemetry_string = f"{int(time.time() * 1000)},{altitude},{temp},{pressure},50.0,4.10,0.0,0.0,1.0,0.0,0.0,0.0,22.57,88.36"

                await websocket.send(telemetry_string)
                await asyncio.sleep(0.01)

            end_time = time.time()
            total_time = end_time - start_time
            print("WebSocket Stress Test Complete!")
            print("Packets sent: 500")
            print(f"Total time taken: {total_time:.2f} seconds")
            print(f"Throughput: {500 / total_time:.1f} packets/second")

    except Exception as e:
        print(f"WebSocket connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(stress_test())
