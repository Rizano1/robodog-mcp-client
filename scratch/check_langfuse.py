from langfuse import get_client
client = get_client()
print(f"Has get_current_trace_id: {hasattr(client, 'get_current_trace_id')}")
print(f"Has get_current_observation_id: {hasattr(client, 'get_current_observation_id')}")
