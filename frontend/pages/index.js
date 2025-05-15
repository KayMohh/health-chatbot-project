import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post("/api/query", { question });
      setAnswer(res.data.answer);
    } catch (err) {
      setAnswer("Error fetching response.");
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: 20 }}>
      <h1>RAG Chatbot</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Ask something..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{ width: "100%", padding: 10 }}
        />
        <button type="submit" disabled={loading} style={{ marginTop: 10, padding: 10 }}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>

      {answer && (
        <div style={{ marginTop: 20, whiteSpace: "pre-wrap" }}>
          <strong>Bot:</strong> {answer}
        </div>
      )}
    </div>
  );
}