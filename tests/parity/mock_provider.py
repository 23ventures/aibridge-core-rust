"""Fake OpenAI, Anthropic and Gemini endpoints for parity tests and benchmarks (no real provider calls).

    python tests/parity/mock_provider.py 9100

Accepts only the key "sk-test-good-key-1234". Replies echo what the adapter sent (model, system
prompt, turns, temperature, max tokens) so request translation differences show up in the output.

The last user message selects a scenario:
    "please fail"   429 before streaming          "fail mid"     error event after one delta
    "server error"  500 with an HTML body          "bad json"     200 with a non-JSON body
    "unicode"       reply with accents and emoji   "echo key"     error message containing the key
Model "missing-model" answers 404.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

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


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1] if len(sys.argv) > 1 else 9100)), Handler).serve_forever()
