"""Fake OpenAI, Anthropic and Gemini endpoints for parity tests and benchmarks (no real provider calls).

    python tests/parity/mock_provider.py 9100

Accepts only the key "sk-test-good-key-1234". Replies echo what the adapter sent (model, system
prompt, turns, temperature, max tokens) so request translation differences show up in the output.

The last user message selects a scenario:
    "please fail"   429 before streaming          "fail mid"     error event after one delta
    "server error"  500 with an HTML body          "bad json"     200 with a non-JSON body
    "unicode"       reply with accents and emoji   "echo key"     error message containing the key
Model "missing-model" answers 404.

Recording mode: any path under ``/rec/<provider>/...`` answers in that path's protocol (OpenAI,
Anthropic, Gemini, Cohere, DashScope, Responses, Ollama) with a reply whose text is the exact
request it received (path, query, relevant headers, body), so comparing replies compares requests.
When the request asks for structured output the echo comes back the way that provider returns
it (a tool call, a function call), and replies carry cache usage, so parsing is exercised too.
Model "thinker" puts a ``<think>`` block before the text.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, parse_qsl, urlparse

GOOD_KEY = "sk-test-good-key-1234"
OPENAI_MODELS = ["mock-small", "mock-large", "text-embedding-mock", "whisper-mock"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    # ---------------------------------------------------------------- helpers
    def _key(self):
        auth = self.headers.get("Authorization", "")
        return self.headers.get("x-api-key") or self.headers.get("x-goog-api-key") or auth.removeprefix("Bearer ")

    def _raw(self, status, body, content_type):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, payload):
        self._raw(status, json.dumps(payload), "application/json")

    def _start_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _send(self, data, event=None):
        prefix = f"event: {event}\n" if event else ""
        self.wfile.write(f"{prefix}data: {data}\n\n".encode())
        self.wfile.flush()

    def _unauthorized(self):
        self._json(401, {"error": {"message": "Incorrect API key provided", "type": "authentication_error"}})

    def _scenario(self, last, stream_capable=True):
        """Returns True when the scenario already answered."""
        if last == "please fail":
            self._json(429, {"error": {"message": "Rate limit reached for mock", "type": "rate_limit"}})
            return True
        if last == "server error":
            self._raw(500, "<html><body>Internal Server Error</body></html>", "text/html")
            return True
        if last == "echo key":
            self._json(400, {"error": {"message": f"key {self._key()} is not allowed here"}})
            return True
        return False

    @staticmethod
    def _reply(model, system, turns, extra=""):
        last = turns[-1]
        if last == "unicode":
            return "Café ☕ naïve — 日本語 😀"
        return f"Echo from **{model}**: {last}. (turns={len(turns)}, system={system or 'none'}{extra})"

    # ---------------------------------------------------------------- routing
    def do_GET(self):
        if self.path.startswith("/rec/"):
            return self._record("GET", None)
        if self._key() != GOOD_KEY:
            return self._unauthorized()
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/v1/models":
            if "anthropic-version" in self.headers:
                if query.get("after_id") == ["claude-a"]:
                    return self._json(200, {"data": [{"id": "claude-b", "display_name": "Claude B"}], "has_more": False})
                return self._json(200, {"data": [{"id": "claude-a", "display_name": "Claude A"}],
                                        "has_more": True, "last_id": "claude-a"})
            return self._json(200, {"data": [{"id": m, "context_length": 8192} for m in OPENAI_MODELS]})
        if url.path == "/v1beta/models":
            if query.get("pageToken") == ["p2"]:
                return self._json(200, {"models": [
                    {"name": "models/gemini-b", "displayName": "Gemini B", "inputTokenLimit": 1000,
                     "supportedGenerationMethods": ["generateContent"]}]})
            return self._json(200, {"nextPageToken": "p2", "models": [
                {"name": "models/gemini-a", "supportedGenerationMethods": ["generateContent", "countTokens"]},
                {"name": "models/embedding-x", "supportedGenerationMethods": ["embedContent"]}]})
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path.startswith("/rec/"):
            return self._record("POST", body)
        if self._key() != GOOD_KEY:
            return self._unauthorized()
        url = urlparse(self.path)
        if url.path == "/v1/chat/completions":
            return self._openai(body)
        if url.path == "/v1/messages":
            return self._anthropic(body)
        if url.path.startswith("/v1beta/models/"):
            model, _, method = url.path.removeprefix("/v1beta/models/").partition(":")
            return self._gemini(body, model, method)
        self._json(404, {"error": {"message": "not found"}})

    # ---------------------------------------------------------------- OpenAI
    def _openai(self, body):
        model = body["model"]
        system = next((m["content"] for m in body["messages"] if m["role"] == "system"), None)
        turns = [m["content"] for m in body["messages"] if m["role"] != "system"]
        if model == "missing-model":
            return self._json(404, {"error": {"message": f"The model `{model}` does not exist"}})
        if self._scenario(turns[-1]):
            return
        extra = f", t={body.get('temperature')}, max={body.get('max_completion_tokens') or body.get('max_tokens')}"
        text = self._reply(model, system, turns, extra)
        if not body.get("stream"):
            if turns[-1] == "bad json":
                return self._raw(200, "not json at all", "text/plain")
            return self._json(200, {"model": model + "-0001", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": len(text.split())}})
        self._start_stream()
        words = [w + " " for w in text.split(" ")]
        if turns[-1] == "fail mid":
            self._send(json.dumps({"model": model, "choices": [{"index": 0, "delta": {"content": "par"}}]}))
            self._send(json.dumps({"error": {"message": "overloaded", "type": "server_error"}}))
            return
        for word in words:
            self._send(json.dumps({"model": model, "choices": [{"index": 0, "delta": {"content": word}}]}))
        self._send(json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
        if body.get("stream_options", {}).get("include_usage"):
            self._send(json.dumps({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": len(words)}}))
        self._send("[DONE]")

    # ---------------------------------------------------------------- Anthropic
    def _anthropic(self, body):
        model = body["model"]
        turns = [m["content"] for m in body["messages"]]
        if self._scenario(turns[-1]):
            return
        extra = f", t={body.get('temperature')}, max={body.get('max_tokens')}"
        text = self._reply(model, body.get("system"), turns, extra)
        if not body.get("stream"):
            return self._json(200, {"type": "message", "model": model, "stop_reason": "end_turn",
                                    "content": [{"type": "text", "text": text}],
                                    "usage": {"input_tokens": 13, "output_tokens": len(text.split())}})
        self._start_stream()
        self._send(json.dumps({"type": "message_start", "message": {"model": model,
                                                                  "usage": {"input_tokens": 13, "output_tokens": 1}}}),
                   "message_start")
        if turns[-1] == "fail mid":
            self._send(json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "par"}}),
                       "content_block_delta")
            self._send(json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
                       "error")
            return
        words = [w + " " for w in text.split(" ")]
        for word in words:
            self._send(json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": word}}),
                       "content_block_delta")
        self._send(json.dumps({"type": "message_delta", "delta": {"stop_reason": "max_tokens"},
                               "usage": {"output_tokens": len(words)}}), "message_delta")
        self._send(json.dumps({"type": "message_stop"}), "message_stop")

    # ---------------------------------------------------------------- Gemini
    def _gemini(self, body, model, method):
        turns = [c["parts"][0]["text"] for c in body["contents"]]
        system = (body.get("systemInstruction") or {}).get("parts", [{}])[0].get("text")
        if self._scenario(turns[-1]):
            return
        config = body.get("generationConfig", {})
        extra = f", roles={','.join(c['role'] for c in body['contents'])}, t={config.get('temperature')}, " \
                f"max={config.get('maxOutputTokens')}"
        text = self._reply(model, system, turns, extra)
        usage = {"promptTokenCount": 9, "candidatesTokenCount": len(text.split()), "thoughtsTokenCount": 2}
        if method == "generateContent":
            return self._json(200, {"modelVersion": model + "-001", "usageMetadata": usage, "candidates": [
                {"content": {"parts": [{"text": "thinking", "thought": True}, {"text": text}]}, "finishReason": "STOP"}]})
        self._start_stream()
        words = [w + " " for w in text.split(" ")]
        for word in words:
            self._send(json.dumps({"candidates": [{"content": {"parts": [{"text": word}]}}]}))
        self._send(json.dumps({"modelVersion": model + "-001", "usageMetadata": usage,
                               "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}))

    # ---------------------------------------------------------------- recording mode
    ECHO_HEADERS = ("authorization", "x-api-key", "anthropic-version", "x-goog-api-key", "content-type", "accept",
                    "x-dashscope-sse")

    def _record(self, method, body):
        url = urlparse(self.path)
        _, _, provider, rest = url.path.split("/", 3)
        path = "/" + rest
        # the key is shown as a marker, so echoed requests never carry it into responses
        headers = {name: self.headers.get(name).replace(GOOD_KEY, "<good-key>") for name in self.ECHO_HEADERS
                   if self.headers.get(name) is not None}
        echo = json.dumps({"method": method, "path": path, "query": sorted(parse_qsl(url.query)),
                           "headers": headers, "body": body}, ensure_ascii=False)
        if method == "GET":
            return self._json(200, self._models_reply(path, echo))
        model = (body or {}).get("model") or path.split("/models/")[-1].split(":")[0]
        text = ("<think>hidden plan</think>" + echo) if model == "thinker" else echo
        stream = bool(body.get("stream")) or path.endswith(":streamGenerateContent")
        if path.endswith("/chat/completions"):
            return self._rec_openai(body, text, stream)
        if path.endswith("/messages"):
            return self._rec_anthropic(body, text, stream)
        if ":" in path and "/models/" in path:
            return self._rec_gemini(text, stream)
        if path.endswith("/v2/chat"):
            return self._rec_cohere(text, stream)
        if path.endswith("/text-generation/generation"):
            return self._rec_dashscope(body, text, stream)
        if path.endswith("/responses"):
            return self._rec_responses(body, text, stream)
        if path.endswith("/api/chat"):
            return self._rec_ollama(text, stream)
        self._json(404, {"error": {"message": f"no protocol for {path}"}})

    @staticmethod
    def _models_reply(path, echo):
        if path.endswith("/api/tags"):
            return {"models": [{"name": "b:latest", "model": "b:latest"}, {"name": echo}]}
        # one body that every list protocol can read: "data" (OpenAI, Anthropic, Responses) and
        # "models" (Gemini, Cohere)
        return {"data": [{"id": echo, "display_name": "Echo", "context_length": 4096}], "has_more": False,
                "models": [{"name": "models/" + echo, "supportedGenerationMethods": ["generateContent"],
                            "inputTokenLimit": 10, "context_length": 4096}]}

    @staticmethod
    def _chunks(text, size=37):
        return [text[i:i + size] for i in range(0, len(text), size)] or [""]

    def _rec_openai(self, body, text, stream):
        tool = bool(body.get("tools"))
        usage = {"prompt_tokens": 40, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 32}}
        if not stream:
            message = {"role": "assistant", "content": None if tool else text}
            if tool:
                message["tool_calls"] = [{"id": "c1", "type": "function",
                                          "function": {"name": "f", "arguments": text}}]
            return self._json(200, {"model": body["model"] + "-v", "usage": usage, "choices": [
                {"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}]})
        self._start_stream()
        for chunk in self._chunks(text):
            delta = ({"tool_calls": [{"index": 0, "function": {"arguments": chunk}}]} if tool
                     else {"content": chunk})
            self._send(json.dumps({"model": body["model"] + "-v", "choices": [{"index": 0, "delta": delta}]}))
        self._send(json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool else "stop"}]}))
        self._send(json.dumps({"choices": [], "usage": usage}))
        self._send("[DONE]")

    def _rec_anthropic(self, body, text, stream):
        tool = bool(body.get("tools"))
        usage = {"input_tokens": 5, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 300,
                 "output_tokens": 1}
        if not stream:
            content = ([{"type": "text", "text": ""}, {"type": "tool_use", "id": "t", "name": "x",
                                                       "input": {"echo": text, "ünï": [1, 2.5, None]}}]
                       if tool else [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": text}])
            return self._json(200, {"type": "message", "model": body["model"], "content": content,
                                    "stop_reason": "tool_use" if tool else "end_turn",
                                    "usage": {**usage, "output_tokens": 12}})
        self._start_stream()
        self._send(json.dumps({"type": "message_start", "message": {"model": body["model"], "usage": usage}}),
                   "message_start")
        for chunk in self._chunks(text):
            delta = ({"type": "input_json_delta", "partial_json": chunk} if tool
                     else {"type": "text_delta", "text": chunk})
            self._send(json.dumps({"type": "content_block_delta", "index": 0, "delta": delta}), "content_block_delta")
        self._send(json.dumps({"type": "message_delta", "delta": {"stop_reason": "max_tokens"},
                               "usage": {"output_tokens": 21}}), "message_delta")
        self._send(json.dumps({"type": "message_stop"}), "message_stop")

    def _rec_gemini(self, text, stream):
        usage = {"promptTokenCount": 9, "candidatesTokenCount": 4, "thoughtsTokenCount": 2}
        if not stream:
            return self._json(200, {"modelVersion": "g-001", "usageMetadata": usage, "candidates": [
                {"content": {"parts": [{"text": "plan", "thought": True}, {"text": text}]}, "finishReason": "STOP"}]})
        self._start_stream()
        for chunk in self._chunks(text):
            self._send(json.dumps({"candidates": [{"content": {"parts": [{"text": chunk}]}}]}))
        self._send(json.dumps({"modelVersion": "g-001", "usageMetadata": usage,
                               "candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}))

    def _rec_cohere(self, text, stream):
        if not stream:
            return self._json(200, {"id": "r", "finish_reason": "COMPLETE", "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "x"}, {"type": "text", "text": text}]},
                "usage": {"billed_units": {"input_tokens": 3, "output_tokens": 4}, "tokens": {"input_tokens": 30,
                                                                                            "output_tokens": 40}}})
        self._start_stream()
        self._send(json.dumps({"type": "message-start", "id": "r"}), "message-start")
        for chunk in self._chunks(text):
            self._send(json.dumps({"type": "content-delta", "index": 0,
                                   "delta": {"message": {"content": {"text": chunk}}}}), "content-delta")
        self._send(json.dumps({"type": "message-end", "delta": {"finish_reason": "MAX_TOKENS",
                                                                "usage": {"tokens": {"input_tokens": 7,
                                                                                     "output_tokens": 8}}}}),
                   "message-end")

    def _rec_dashscope(self, body, text, stream):
        tool = (body.get("parameters") or {}).get("response_format") is not None and body["model"] == "qwen-tool"
        def choice(chunk, finish):
            message = {"role": "assistant", "content": "" if tool else chunk}
            if tool:
                message["tool_calls"] = [{"function": {"name": "f", "arguments": chunk}}]
            return {"output": {"choices": [{"message": message, "finish_reason": finish}]}}
        if not stream:
            return self._json(200, {**choice(text, "stop"), "usage": {"input_tokens": 6, "output_tokens": 7},
                                    "request_id": "q"})
        self._start_stream()
        chunks = self._chunks(text)
        for index, chunk in enumerate(chunks):
            last = index == len(chunks) - 1
            payload = choice(chunk, "stop" if last else "null")
            if last:
                payload["usage"] = {"input_tokens": 6, "output_tokens": 7}
            self._send(json.dumps(payload), "result")

    def _rec_responses(self, body, text, stream):
        tool = bool(body.get("tools"))
        usage = {"input_tokens": 50, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 20}}
        output = ([{"type": "reasoning", "summary": []},
                   {"type": "function_call", "name": "f", "call_id": "c", "arguments": text}] if tool
                  else [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}])
        response = {"id": "r", "model": body["model"] + "-v", "status": "completed", "output": output, "usage": usage}
        if not stream:
            return self._json(200, response)
        self._start_stream()
        kind = "response.function_call_arguments.delta" if tool else "response.output_text.delta"
        for chunk in self._chunks(text):
            self._send(json.dumps({"type": kind, "delta": chunk}), kind)
        self._send(json.dumps({"type": "response.completed", "response": response}), "response.completed")

    def _rec_ollama(self, text, stream):
        final = {"model": "llama-v", "done": True, "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 3}
        if not stream:
            return self._json(200, {**final, "message": {"role": "assistant", "content": text}})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for chunk in self._chunks(text):
            self.wfile.write((json.dumps({"model": "llama-v", "message": {"content": chunk}, "done": False}) + "\n").encode())
            self.wfile.flush()
        self.wfile.write(json.dumps({**final, "message": {"content": ""}}).encode())  # no trailing newline
        self.wfile.flush()


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1] if len(sys.argv) > 1 else 9100)), Handler).serve_forever()
