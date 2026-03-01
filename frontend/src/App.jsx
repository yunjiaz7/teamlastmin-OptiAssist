import { useMemo, useState } from "react";
import { sendChatMessage, toAbsoluteAssetUrl, uploadImage } from "./api";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [session, setSession] = useState(null);
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState([]);
  const [lastResult, setLastResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  const canSend = useMemo(() => Boolean(session?.session_id) && message.trim().length > 0 && !sending, [session, message, sending]);

  async function handleUpload(event) {
    event.preventDefault();
    setError("");
    if (!selectedFile) {
      setError("Please choose an image first.");
      return;
    }
    setUploading(true);
    try {
      const data = await uploadImage(selectedFile);
      setSession(data);
      setHistory([]);
      setLastResult(null);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setUploading(false);
    }
  }

  async function handleSendMessage(event) {
    event.preventDefault();
    if (!canSend) return;
    setSending(true);
    setError("");
    try {
      const data = await sendChatMessage(session.session_id, message.trim());
      setHistory(data.history || []);
      setLastResult(data);
      setMessage("");
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="app-shell">
      <h1>Fundus Agent Console</h1>

      <section className="panel">
        <h2>1) Upload Image</h2>
        <form onSubmit={handleUpload} className="upload-form">
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
            disabled={uploading}
          />
          <button type="submit" disabled={uploading}>
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </form>
        {session && (
          <div className="upload-meta">
            <p><strong>Session:</strong> {session.session_id}</p>
            <p><strong>File:</strong> {session.filename}</p>
            <img src={toAbsoluteAssetUrl(session.image_url)} alt="Uploaded fundus" className="preview-image" />
          </div>
        )}
      </section>

      <section className="panel">
        <h2>2) Chat With Agent</h2>
        <form onSubmit={handleSendMessage} className="chat-form">
          <textarea
            rows={4}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder={session ? "Ask about this uploaded image..." : "Upload an image to start."}
            disabled={!session || sending}
          />
          <button type="submit" disabled={!canSend}>
            {sending ? "Sending..." : "Send"}
          </button>
        </form>

        <div className="history">
          {history.length === 0 && <p>No conversation yet.</p>}
          {history.map((turn, index) => (
            <div key={`${turn.timestamp}-${index}`} className={`message ${turn.role}`}>
              <p className="message-role">{turn.role}</p>
              <p>{turn.content}</p>
            </div>
          ))}
        </div>
      </section>

      {lastResult && (
        <section className="panel">
          <h2>Latest Trace</h2>
          <p><strong>Confidence:</strong> {lastResult.confidence}</p>
          <p><strong>Tools Called:</strong> {(lastResult.decision_path?.tools_called || []).join(", ") || "none"}</p>
          {Array.isArray(lastResult.errors) && lastResult.errors.length > 0 && (
            <div>
              <strong>Errors</strong>
              <ul>
                {lastResult.errors.map((item, idx) => (
                  <li key={`${idx}-${item}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default App;
