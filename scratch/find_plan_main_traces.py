import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

dotenv_path = "/home/icar/ano/prata/.env"
load_dotenv(dotenv_path)

base_url = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3001")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")

auth = HTTPBasicAuth(public_key, secret_key)

# Fetch traces
traces = []
page = 1
while True:
    url = f"{base_url.rstrip('/')}/api/public/traces"
    resp = requests.get(url, auth=auth, params={"page": page, "limit": 100})
    if resp.status_code != 200:
        break
    data = resp.json().get("data", [])
    if not data:
        break
    traces.extend(data)
    meta = resp.json().get("meta", {})
    if page >= meta.get("totalPages", 1):
        break
    page += 1

print(f"Fetched {len(traces)} traces in total.")

planning_keywords = [
    "inspeksi", "apar", "stop kontak", "tong sampah", "lantai 9", "current_coordinates"
]

matching_main_traces = []
for trace in traces:
    if trace.get("name") in ["main", "process_chat"]:
        # check if input or output has planning words
        inp_str = str(trace.get("input") or "")
        out_str = str(trace.get("output") or "")
        
        # Check if it matches any plan prompt characteristic
        is_plan_eval = False
        if "plan_" in inp_str or "plan_" in out_str:
            is_plan_eval = True
        elif "current_coordinates" in inp_str and ("apar" in inp_str or "tong sampah" in inp_str or "stop kontak" in inp_str):
            is_plan_eval = True
            
        if is_plan_eval:
            matching_main_traces.append(trace)

print(f"Found {len(matching_main_traces)} main/process_chat traces that look like planning evaluation.")
for idx, trace in enumerate(matching_main_traces[:15]):
    print(f"[{idx+1}] Trace ID: {trace.get('id')}")
    print(f"    Name: {trace.get('name')}")
    print(f"    Tags: {trace.get('tags')}")
    print(f"    Session ID: {trace.get('sessionId')}")
    print(f"    Input snippet: {str(trace.get('input'))[:150]}")
    print("-" * 50)
