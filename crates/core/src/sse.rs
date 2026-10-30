//! Incremental Server-Sent Events decoder: bytes in, `(event, data)` pairs out.
//!
//! Works on bytes so a multi-byte UTF-8 character split across network chunks is never
//! mangled: lines are only cut at ASCII `\r` / `\n`, then decoded.

#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

pub type SseEvent = (Option<String>, String);

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut start = 0;
        while let Some((index, width)) = find_newline(&self.buffer[start..]) {
            let line_end = start + index;
            let line = String::from_utf8_lossy(&self.buffer[start..line_end]).into_owned();
            start = line_end + width;
            if let Some(event) = self.line(&line) {
                events.push(event);
            }
        }
        if start > 0 {
            self.buffer.drain(..start);
        }
        events
    }

    /// Emit whatever is left once the stream ends.
    pub fn flush(&mut self) -> Vec<SseEvent> {
        let mut events = if self.buffer.is_empty() { Vec::new() } else { self.feed(b"\n") };
        if let Some(event) = self.line("") {
            events.push(event);
        }
        events
    }

    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            if self.data.is_empty() {
                self.event = None;
                return None;
            }
            let data = std::mem::take(&mut self.data).join("\n");
            return Some((self.event.take(), data));
        }
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = match line.find(':') {
            Some(index) => (&line[..index], &line[index + 1..]),
            None => (line, ""),
        };
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => self.event = Some(value.to_owned()),
            "data" => self.data.push(value.to_owned()),
            _ => {}
        }
        None
    }
}

fn find_newline(bytes: &[u8]) -> Option<(usize, usize)> {
    for (index, &byte) in bytes.iter().enumerate() {
        if byte == b'\n' {
            return Some((index, 1));
        }
        if byte == b'\r' {
            return match bytes.get(index + 1) {
                Some(b'\n') => Some((index, 2)),
                Some(_) => Some((index, 1)),
                None => None, // wait: may be the first half of \r\n
            };
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_split_chunks_and_line_endings() {
        let mut decoder = SseDecoder::new();
        let mut out = decoder.feed(b"event: a\r");
        out.extend(decoder.feed(b"\ndata: 1\r\n\r\ndata: x"));
        out.extend(decoder.feed(b"\ndata: y\n\n: comment\n"));
        out.extend(decoder.feed(b"data:z"));
        out.extend(decoder.flush());
        assert_eq!(out, vec![(Some("a".into()), "1".into()), (None, "x\ny".into()), (None, "z".into())]);
    }

    #[test]
    fn multibyte_characters_split_across_chunks() {
        let bytes = "data: héllo\n\n".as_bytes();
        let mut decoder = SseDecoder::new();
        let mut out = decoder.feed(&bytes[..8]);
        out.extend(decoder.feed(&bytes[8..]));
        assert_eq!(out, vec![(None, "héllo".into())]);
    }
}
