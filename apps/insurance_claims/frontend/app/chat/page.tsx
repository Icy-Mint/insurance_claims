"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type Message = {
  role: "user" | "assistant";
  text: string;
};

type ChatResponse = {
  reply: string;
  phase: string;
  ui_hints: { quick_replies?: string[] };
  state: Record<string, unknown>;
};

const PHASE_LABELS: Record<string, string> = {
  VERIFY_ID: "Verifying identity",
  RESOLVE_INTENT: "Understanding your request",
  PROCESS_CASE: "Looking into your claim",
  POST_PROCESS: "Wrapping up",
};

function newSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text: "Hi, thanks for calling insurance claims support. To get started, could you tell me your full name?",
    },
  ]);
  const [input, setInput] = useState("");
  const [quickReplies, setQuickReplies] = useState<string[]>([]);
  const [phase, setPhase] = useState("VERIFY_ID");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSessionId(newSessionId());
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed || !sessionId || loading) return;

    setMessages((m) => [...m, { role: "user", text: trimmed }]);
    setInput("");
    setQuickReplies([]);
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: trimmed }),
      });
      if (!res.ok) throw new Error(`Server error (${res.status})`);
      const data: ChatResponse = await res.json();
      setMessages((m) => [...m, { role: "assistant", text: data.reply }]);
      setQuickReplies(data.ui_hints?.quick_replies || []);
      setPhase(data.phase);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={styles.page}>
      <div style={styles.card}>
        <header style={styles.header}>
          <div>
            <h1 style={styles.title}>Insurance Claims Support</h1>
            <span style={styles.phaseBadge}>{PHASE_LABELS[phase] || phase}</span>
          </div>
        </header>

        <div style={styles.messages}>
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                ...styles.bubbleRow,
                justifyContent: m.role === "user" ? "flex-end" : "flex-start",
              }}
            >
              <div style={m.role === "user" ? styles.userBubble : styles.assistantBubble}>{m.text}</div>
            </div>
          ))}
          {loading && (
            <div style={styles.bubbleRow}>
              <div style={styles.assistantBubble}>…</div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {error && <div style={styles.error}>{error}</div>}

        {quickReplies.length > 0 && (
          <div style={styles.quickReplies}>
            {quickReplies.map((qr) => (
              <button key={qr} style={styles.quickReplyButton} onClick={() => send(qr)} disabled={loading}>
                {qr}
              </button>
            ))}
          </div>
        )}

        <form
          style={styles.inputRow}
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            style={styles.input}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            disabled={loading}
          />
          <button type="submit" style={styles.sendButton} disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: 16,
  },
  card: {
    width: "100%",
    maxWidth: 640,
    height: "85vh",
    background: "#fff",
    borderRadius: 12,
    boxShadow: "0 4px 24px rgba(0,0,0,0.08)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  header: {
    padding: "16px 20px",
    borderBottom: "1px solid #eee",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  title: { fontSize: 18, margin: 0 },
  phaseBadge: {
    fontSize: 12,
    color: "#555",
    background: "#eef1f5",
    borderRadius: 999,
    padding: "4px 10px",
    marginTop: 6,
    display: "inline-block",
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    padding: 20,
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  bubbleRow: { display: "flex" },
  userBubble: {
    background: "#2563eb",
    color: "#fff",
    padding: "10px 14px",
    borderRadius: "16px 16px 4px 16px",
    maxWidth: "75%",
    whiteSpace: "pre-wrap",
  },
  assistantBubble: {
    background: "#f1f2f4",
    color: "#1a1a1a",
    padding: "10px 14px",
    borderRadius: "16px 16px 16px 4px",
    maxWidth: "75%",
    whiteSpace: "pre-wrap",
  },
  quickReplies: {
    display: "flex",
    flexWrap: "wrap",
    gap: 8,
    padding: "0 20px 12px",
  },
  quickReplyButton: {
    border: "1px solid #2563eb",
    color: "#2563eb",
    background: "#fff",
    borderRadius: 999,
    padding: "6px 14px",
    fontSize: 13,
    cursor: "pointer",
  },
  inputRow: {
    display: "flex",
    gap: 8,
    padding: 16,
    borderTop: "1px solid #eee",
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    borderRadius: 8,
    border: "1px solid #ddd",
    fontSize: 14,
  },
  sendButton: {
    padding: "10px 18px",
    borderRadius: 8,
    border: "none",
    background: "#2563eb",
    color: "#fff",
    fontSize: 14,
    cursor: "pointer",
  },
  error: {
    color: "#b91c1c",
    fontSize: 13,
    padding: "0 20px 8px",
  },
};
