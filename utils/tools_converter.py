from google.genai.types import Tool, FunctionDeclaration

def clean_schema(schema):
    """
    Recursively removes unsupported fields from the JSON schema for Gemini.

    Args:
        schema (dict): The schema dictionary.

    Returns:
        dict: Cleaned schema without 'title' and 'additionalProperties' fields.
    """
    if isinstance(schema, dict):
        schema.pop("title", None)
        schema.pop("additionalProperties", None)
        schema.pop("additional_properties", None)

        # Recursively clean nested properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            for key in schema["properties"]:
                schema["properties"][key] = clean_schema(schema["properties"][key])
        
        # Also clean items for arrays
        if "items" in schema:
            schema["items"] = clean_schema(schema["items"])

    return schema

def convert_mcp_tools_to_gemini(mcp_tools):
    """
    Converts MCP tool definitions to the correct format for Gemini API function calling.

    Args:
        mcp_tools (list): List of MCP tool objects with 'name', 'description', and 'inputSchema'.

    Returns:
        list: List of Gemini Tool objects with properly formatted function declarations.
    """
    gemini_tools = []

    for tool in mcp_tools:
        # Ensure inputSchema is a valid JSON schema and clean it
        parameters = clean_schema(tool.inputSchema)

        # Construct the function declaration
        function_declaration = FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=parameters  # Now correctly formatted
        )

        # Wrap in a Tool object
        gemini_tool = Tool(function_declarations=[function_declaration])
        gemini_tools.append(gemini_tool)

    return gemini_tools


def convert_mcp_tools_to_ollama(mcp_tools):
    """
    Converts MCP tool definitions to OpenAI-compatible format for Ollama API.

    Args:
        mcp_tools (list): List of MCP tool objects with 'name', 'description', and 'inputSchema'.

    Returns:
        list: List of OpenAI-compatible tool dicts for Ollama's /v1/chat/completions endpoint.
    """
    ollama_tools = []

    for tool in mcp_tools:
        # Deep copy and clean the schema
        parameters = clean_schema(dict(tool.inputSchema))

        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters
            }
        })

    return ollama_tools