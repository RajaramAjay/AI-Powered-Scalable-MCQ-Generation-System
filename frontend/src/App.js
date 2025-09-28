// frontend/src/App.js

import React, { useState, useEffect } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first!");
    setQuestions([]);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Initiate SSE connection
      const eventSource = new EventSource(
        `http://localhost:8000/upload_sse?filename=${encodeURIComponent(
          file.name
        )}`
      );

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setQuestions((prev) => [...prev, data]);
      };

      eventSource.onerror = (err) => {
        console.error("SSE error:", err);
        eventSource.close();
        setLoading(false);
      };

      // Upload file
      await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">LLM Question Paper Generator</h1>

      <input type="file" onChange={handleFileChange} />
      <button
        onClick={handleUpload}
        className="ml-2 px-4 py-2 bg-blue-500 text-white rounded"
      >
        Generate Questions
      </button>

      {loading && <p className="mt-4 text-gray-500">Generating questions...</p>}

      <div className="mt-6 space-y-4">
        {questions.map((q, idx) => (
          <div
            key={idx}
            className="border p-4 rounded shadow bg-gray-50"
          >
            <p className="font-semibold">
              Q{idx + 1}: {q.question}
            </p>
            <ul className="list-disc ml-5 mt-2">
              {Object.entries(q.choices).map(([key, choice]) => (
                <li key={key}>
                  <strong>{key}:</strong> {choice}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;

