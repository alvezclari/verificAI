import { useState } from "react";
import Home from "./components/Home.jsx";
import Result from "./components/Result.jsx";

export default function App() {
  const [result, setResult] = useState(null);
  return (
    <>
      <header>
        <div className="brand"><div className="logo">L</div>VerificAI</div>
      </header>
      <main>
        {result ? (
          <Result data={result} onBack={() => setResult(null)} />
        ) : (
          <Home onResult={setResult} />
        )}
      </main>
    </>
  );
}
