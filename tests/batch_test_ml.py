"""
Batch Testing Script for Cognitive CanSat Machine Learning API
Sends a batch of 100 mock telemetry requests to /api/predict to benchmark latency and throughput.
Supports both 'requests' and standard library 'urllib.request'.
"""

import time
import random
import json

try:
    import requests
except ImportError:
    requests = None

import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000/api/predict"

def send_payload(payload):
    if requests is not None:
        resp = requests.post(API_URL, json=payload, timeout=5)
        return resp.status_code, resp.text
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")
        except Exception as e:
            raise e

def main():
    batch_data = []
    for _ in range(100):
        batch_data.append({
            "temp": round(random.uniform(20.0, 35.0), 2),
            "pressure": round(random.uniform(900.0, 1013.25), 2),
            "altitude": round(random.uniform(0.0, 500.0), 2)
        })

    print(f"Starting ML API batch test with {len(batch_data)} records...")
    start_time = time.time()
    success_count = 0

    for data in batch_data:
        try:
            status_code, text = send_payload(data)
            if status_code == 200:
                success_count += 1
            else:
                print(f"Server rejected data: HTTP {status_code} - {text}")
                break
        except Exception as e:
            print(f"Request failed: {e}")
            break

    end_time = time.time()
    total_time = end_time - start_time

    print("Batch Test Complete!")
    print(f"Successful predictions: {success_count}/{len(batch_data)}")
    print(f"Total time taken: {total_time:.2f} seconds")
    if success_count > 0:
        print(f"Average latency per inference: {(total_time / success_count) * 1000:.2f} ms")

if __name__ == "__main__":
    main()
