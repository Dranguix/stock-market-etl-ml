"""Chat con tools contra la API de Anthropic.

El ciclo del inversor habla el formato de chat completions (tool_calls).
Aquí se traduce a la API de Messages y la respuesta se devuelve en ese mismo formato.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-5-5"


def function_tool(schema: dict) -> dict:
    """Pasa el tool_schema() al formato function de chat completions."""
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        return schema
    parameters = schema.get("input_schema") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description") or "",
            "parameters": parameters,
        },
    }


def _anthropic_tool(tool: dict) -> dict:
    fn = tool["function"] if tool.get("type") == "function" else tool
    parameters = fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
    return {
        "name": fn["name"],
        "description": fn.get("description") or "",
        "input_schema": parameters,
    }


def _to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    system: list[str] = []
    out: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            if message.get("content"):
                system.append(str(message["content"]))
            continue
        if role == "assistant":
            content: list[dict] = []
            text = message.get("content")
            if text:
                content.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                content.append({
                    "type": "tool_use",
                    "id": call.get("id"),
                    "name": fn.get("name"),
                    "input": args if isinstance(args, dict) else {},
                })
            out.append({"role": "assistant", "content": content or ""})
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id"),
                "content": message.get("content") or "",
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue
        out.append({"role": role or "user", "content": message.get("content") or ""})
    return "\n\n".join(system), out


def _from_anthropic(body: dict) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in body.get("content") or []:
        if block.get("type") == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            })
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = "tool_calls" if tool_calls else "stop"
    return {"choices": [{"finish_reason": finish, "message": message}], "model": body.get("model")}


def chat_with_tools(messages: list[dict], tools: list[dict], model: str | None = None) -> dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Falta ANTHROPIC_API_KEY")
    model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    system, converted = _to_anthropic(messages)
    response = httpx.post(
        API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 2500,
            "system": system,
            "tools": [_anthropic_tool(tool) for tool in tools],
            "messages": converted,
        },
        timeout=90,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Anthropic {response.status_code}: {response.text[:500]}")
    return _from_anthropic(response.json())
