//! Request bodies, validated with the same rules (and lax coercions) as the pydantic models of
//! the Python server. Problems are collected as `{"loc": [...], "msg": "..."}` details.

use aibridge_core::{ChatMessage, Role};
use serde_json::{json, Map, Value};

pub type Details = Vec<Value>;

fn detail(loc: &[Value], msg: impl Into<String>) -> Value {
    json!({"loc": loc, "msg": msg.into()})
}

fn loc(path: &[&str]) -> Vec<Value> {
    std::iter::once("body").chain(path.iter().copied()).map(|p| json!(p)).collect()
}

pub struct ChatBody {
    pub provider: String,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub system: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub stream: bool,
    pub base_url: Option<String>,
}

pub struct ModelsBody {
    pub provider: String,
    pub base_url: Option<String>,
}

/// Parse the raw body: invalid JSON or a non-object is a validation error.
pub fn parse_object(raw: &[u8]) -> Result<Map<String, Value>, Details> {
    if raw.iter().all(u8::is_ascii_whitespace) {
        return Err(vec![detail(&loc(&[]), "Field required")]);
    }
    match serde_json::from_slice::<Value>(raw) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err(vec![detail(&loc(&[]), "Input should be a valid dictionary or object to extract fields from")]),
        Err(error) => Err(vec![
            json!({"loc": ["body", char_offset(raw, error.line(), error.column())], "msg": "JSON decode error"}),
        ]),
    }
}

/// 0-based character offset of a serde (line, column) position, like `JSONDecodeError.pos`.
fn char_offset(raw: &[u8], line: usize, column: usize) -> usize {
    let text = String::from_utf8_lossy(raw);
    let before: usize = text.split_inclusive('\n').take(line.saturating_sub(1)).map(|l| l.chars().count()).sum();
    before + column.saturating_sub(1)
}

struct Checker<'a> {
    body: &'a Map<String, Value>,
    errors: Details,
}

impl<'a> Checker<'a> {
    fn string(&mut self, field: &str, required: bool, min: usize, max: usize) -> Option<String> {
        match self.body.get(field) {
            None | Some(Value::Null) if !required => None,
            None => {
                self.errors.push(detail(&loc(&[field]), "Field required"));
                None
            }
            Some(Value::String(text)) => {
                let length = text.chars().count();
                if length < min {
                    let unit = if min == 1 { "character" } else { "characters" };
                    self.errors.push(detail(&loc(&[field]), format!("String should have at least {min} {unit}")));
                    None
                } else if length > max {
                    self.errors.push(detail(&loc(&[field]), format!("String should have at most {max} characters")));
                    None
                } else {
                    Some(text.clone())
                }
            }
            Some(_) => {
                self.errors.push(detail(&loc(&[field]), "Input should be a valid string"));
                None
            }
        }
    }

    fn number(&mut self, field: &str, min: f64, max: f64, integer: bool) -> Option<f64> {
        let value = match self.body.get(field) {
            None | Some(Value::Null) => return None,
            Some(Value::Number(n)) => n.as_f64(),
            Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
            Some(_) => None,
        };
        let kind = if integer { "a valid integer" } else { "a valid number" };
        let Some(value) = value.filter(|v| v.is_finite()) else {
            self.errors.push(detail(&loc(&[field]), format!("Input should be {kind}")));
            return None;
        };
        if integer && value.fract() != 0.0 {
            self.errors
                .push(detail(&loc(&[field]), "Input should be a valid integer, got a number with a fractional part"));
            return None;
        }
        let show = |v: f64| {
            if v.fract() == 0.0 {
                format!("{}", v as i64)
            } else {
                v.to_string()
            }
        };
        if value < min {
            self.errors.push(detail(&loc(&[field]), format!("Input should be greater than or equal to {}", show(min))));
            return None;
        }
        if value > max {
            self.errors.push(detail(&loc(&[field]), format!("Input should be less than or equal to {}", show(max))));
            return None;
        }
        Some(value)
    }

    fn boolean(&mut self, field: &str, default: bool) -> bool {
        let parsed = match self.body.get(field) {
            None => return default,
            Some(Value::Bool(b)) => Some(*b),
            Some(Value::Number(n)) => match n.as_f64() {
                Some(1.0) => Some(true),
                Some(0.0) => Some(false),
                _ => None,
            },
            Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "t" | "yes" | "y" | "on" => Some(true),
                "0" | "false" | "f" | "no" | "n" | "off" => Some(false),
                _ => None,
            },
            Some(_) => None,
        };
        parsed.unwrap_or_else(|| {
            self.errors.push(detail(&loc(&[field]), "Input should be a valid boolean"));
            default
        })
    }

    fn messages(&mut self) -> Vec<ChatMessage> {
        let items = match self.body.get("messages") {
            None => {
                self.errors.push(detail(&loc(&["messages"]), "Field required"));
                return Vec::new();
            }
            Some(Value::Array(items)) => items,
            Some(_) => {
                self.errors.push(detail(&loc(&["messages"]), "Input should be a valid list"));
                return Vec::new();
            }
        };
        if items.is_empty() {
            self.errors.push(detail(&loc(&["messages"]), "List should have at least 1 item after validation, not 0"));
            return Vec::new();
        }
        let mut out = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            let at = |field: &str| -> Vec<Value> {
                let mut path = vec![json!("body"), json!("messages"), json!(index)];
                if !field.is_empty() {
                    path.push(json!(field));
                }
                path
            };
            let Some(object) = item.as_object() else {
                self.errors.push(detail(&at(""), "Input should be a valid dictionary or instance of Message"));
                continue;
            };
            let role = match object.get("role") {
                None => {
                    self.errors.push(detail(&at("role"), "Field required"));
                    None
                }
                Some(value) => {
                    let role = value.as_str().and_then(Role::parse);
                    if role.is_none() {
                        self.errors.push(detail(&at("role"), "Input should be 'system', 'user' or 'assistant'"));
                    }
                    role
                }
            };
            let content = match object.get("content") {
                None => {
                    self.errors.push(detail(&at("content"), "Field required"));
                    None
                }
                Some(Value::String(text)) => Some(text.clone()),
                Some(_) => {
                    self.errors.push(detail(&at("content"), "Input should be a valid string"));
                    None
                }
            };
            if let (Some(role), Some(content)) = (role, content) {
                out.push(ChatMessage::new(role, content));
            }
        }
        out
    }
}

pub fn chat_body(body: &Map<String, Value>) -> Result<ChatBody, Details> {
    let mut check = Checker { body, errors: Vec::new() };
    let provider = check.string("provider", true, 1, 64);
    let model = check.string("model", true, 1, 256);
    let messages = check.messages();
    let system = check.string("system", false, 0, usize::MAX);
    let temperature = check.number("temperature", 0.0, 2.0, false);
    let max_tokens = check.number("max_tokens", 1.0, 1_000_000.0, true).map(|v| v as u64);
    let stream = check.boolean("stream", true);
    let base_url = check.string("base_url", false, 0, 2048);
    if !check.errors.is_empty() {
        return Err(check.errors);
    }
    Ok(ChatBody {
        provider: provider.unwrap_or_default(),
        model: model.unwrap_or_default(),
        messages,
        system,
        temperature,
        max_tokens,
        stream,
        base_url,
    })
}

pub fn models_body(body: &Map<String, Value>) -> Result<ModelsBody, Details> {
    let mut check = Checker { body, errors: Vec::new() };
    let provider = check.string("provider", true, 1, 64);
    let base_url = check.string("base_url", false, 0, 2048);
    if !check.errors.is_empty() {
        return Err(check.errors);
    }
    Ok(ModelsBody { provider: provider.unwrap_or_default(), base_url })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn valid_body_with_lax_coercions() {
        let body = chat_body(&obj(json!({"provider": "openai", "model": "m", "stream": "false", "max_tokens": 5.0,
                                          "temperature": "0.5", "messages": [{"role": "user", "content": "hi"}]})))
        .ok()
        .unwrap();
        assert!(!body.stream);
        assert_eq!((body.max_tokens, body.temperature), (Some(5), Some(0.5)));
    }

    #[test]
    fn collects_every_problem() {
        let errors =
            chat_body(&obj(json!({"provider": "", "messages": [{"role": "robot", "content": 1}], "temperature": 3})))
                .err()
                .unwrap();
        let messages: Vec<&str> = errors.iter().map(|e| e["msg"].as_str().unwrap()).collect();
        assert_eq!(
            messages,
            [
                "String should have at least 1 character",
                "Field required",
                "Input should be 'system', 'user' or 'assistant'",
                "Input should be a valid string",
                "Input should be less than or equal to 2",
            ]
        );
        assert_eq!(errors[2]["loc"], json!(["body", "messages", 0, "role"]));
    }

    #[test]
    fn invalid_json() {
        assert_eq!(parse_object(b"{nope").unwrap_err()[0]["loc"], json!(["body", 1]));
        assert_eq!(parse_object(b"{\n  x").unwrap_err()[0]["loc"], json!(["body", 4]));
        assert!(parse_object(b"[]").is_err());
        assert!(parse_object(b"").is_err());
    }
}
