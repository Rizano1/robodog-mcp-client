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

session_ids = [str(i) for i in range(538, 574)]
print(f"Target Session IDs: {session_ids}")

traces_by_session = {sid: [] for sid in session_ids}
for trace in traces:
    sid = trace.get("sessionId")
    if sid in traces_by_session:
        traces_by_session[sid].append(trace)

for sid in session_ids:
    s_traces = traces_by_session[sid]
    print(f"Session {sid}: Found {len(s_traces)} traces.")
    for t in s_traces:
        print(f"  Trace ID: {t.get('id')} | Name: {t.get('name')} | Tags: {t.get('tags')}")
    print("-" * 50)
