"use client";

import { useRef, useState } from "react";

type Match = {
  surah: number;
  ayah: number;
  text: string;
  cer: number;
  char_similarity: number;
  edit_distance: number;
};

type ApiResult = {
  ok: boolean;
  request_id?: string;
  mode?: "guided" | "autodetect";
  audio_path?: string;
  surah?: number;
  ayah?: number;
  reference?: {
    surah: number;
    ayah: number;
    text: string;
    text_compact: string;
    source_id?: string;
  } | null;
  autodetect?: {
    accepted: boolean;
    needs_confirmation: boolean;
    verdict: string;
    confidence: number;
    pred: string;
    best_match?: Match;
    matches?: Match[];
  };
  content_gate?: {
    accepted: boolean;
    verdict: string;
    mode: string;
    exact: boolean;
    gold: string;
    pred: string;
    gold_compact: string;
    pred_compact: string;
    char_accuracy: number;
    cer: number;
    edit_distance: number;
    gold_len: number;
    pred_len: number;
  } | null;
  tajweed?: unknown;
  message?: string;
  error?: string;
};

export default function Home() {
  const [mode, setMode] = useState<"guided" | "autodetect">("autodetect");
  const [surah, setSurah] = useState("1");
  const [ayah, setAyah] = useState("2");

  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const [result, setResult] = useState<ApiResult | null>(null);
  const [loading, setLoading] = useState(false);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function startRecording() {
    setResult(null);
    setAudioBlob(null);
    setAudioUrl(null);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const recorder = new MediaRecorder(stream);
    recorderRef.current = recorder;
    chunksRef.current = [];

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setAudioBlob(blob);
      setAudioUrl(URL.createObjectURL(blob));
      stream.getTracks().forEach((track) => track.stop());
    };

    recorder.start();
    setRecording(true);
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  async function submitRecording() {
    if (!audioBlob) {
      alert("Record audio first.");
      return;
    }

    setLoading(true);
    setResult(null);

    const form = new FormData();
    form.append("audio", audioBlob, "recording.webm");
    form.append("mode", mode);

    if (mode === "guided") {
      form.append("surah", surah);
      form.append("ayah", ayah);
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/api/assess-recitation", {
        method: "POST",
        body: form,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({
        ok: false,
        error: String(error),
      });
    } finally {
      setLoading(false);
    }
  }

  const gate = result?.content_gate;
  const autodetect = result?.autodetect;

  return (
    <main className="min-h-screen bg-slate-950 p-6 text-white">
      <div className="mx-auto max-w-4xl space-y-6">
        <header className="space-y-2">
          <p className="text-sm uppercase tracking-[0.3em] text-emerald-400">
            Tajweed AI
          </p>
          <h1 className="text-4xl font-bold">Recitation Assessment</h1>
          <p className="max-w-2xl text-slate-300">
            Record your recitation. The system can either check a selected ayah or
            automatically detect the ayah from your voice.
          </p>
        </header>

        <section className="rounded-3xl border border-slate-800 bg-slate-900 p-5 shadow-xl">
          <div className="mb-5 grid gap-3 sm:grid-cols-2">
            <button
              onClick={() => setMode("autodetect")}
              className={`rounded-2xl border p-4 text-left transition ${
                mode === "autodetect"
                  ? "border-emerald-400 bg-emerald-500/10"
                  : "border-slate-700 bg-slate-800"
              }`}
            >
              <div className="text-lg font-semibold">Free Recitation</div>
              <div className="text-sm text-slate-300">
                Record without selecting an ayah. The backend auto-detects it.
              </div>
            </button>

            <button
              onClick={() => setMode("guided")}
              className={`rounded-2xl border p-4 text-left transition ${
                mode === "guided"
                  ? "border-blue-400 bg-blue-500/10"
                  : "border-slate-700 bg-slate-800"
              }`}
            >
              <div className="text-lg font-semibold">Guided Practice</div>
              <div className="text-sm text-slate-300">
                Select the expected surah and ayah before recording.
              </div>
            </button>
          </div>

          {mode === "guided" && (
            <div className="mb-5 grid grid-cols-2 gap-4">
              <label className="space-y-2">
                <span className="text-sm text-slate-300">Surah number</span>
                <input
                  className="w-full rounded-xl bg-slate-800 px-3 py-2 outline-none ring-1 ring-slate-700 focus:ring-blue-400"
                  value={surah}
                  onChange={(e) => setSurah(e.target.value)}
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm text-slate-300">Ayah number</span>
                <input
                  className="w-full rounded-xl bg-slate-800 px-3 py-2 outline-none ring-1 ring-slate-700 focus:ring-blue-400"
                  value={ayah}
                  onChange={(e) => setAyah(e.target.value)}
                />
              </label>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3">
            {!recording ? (
              <button
                onClick={startRecording}
                className="rounded-xl bg-emerald-500 px-5 py-3 font-semibold text-slate-950 hover:bg-emerald-400"
              >
                Start recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="rounded-xl bg-red-500 px-5 py-3 font-semibold text-white hover:bg-red-400"
              >
                Stop recording
              </button>
            )}

            <button
              onClick={submitRecording}
              disabled={!audioBlob || loading}
              className="rounded-xl bg-blue-500 px-5 py-3 font-semibold text-white hover:bg-blue-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? "Assessing..." : "Submit"}
            </button>

            {recording && (
              <span className="animate-pulse text-sm text-red-300">
                Recording...
              </span>
            )}
          </div>

          {audioUrl && (
            <div className="mt-5">
              <p className="mb-2 text-sm text-slate-400">Preview</p>
              <audio controls src={audioUrl} className="w-full" />
            </div>
          )}
        </section>

        {result && (
          <section className="rounded-3xl border border-slate-800 bg-slate-900 p-5 shadow-xl">
            <h2 className="mb-4 text-2xl font-bold">Result</h2>

            {result.error && (
              <pre className="whitespace-pre-wrap rounded-xl bg-red-950 p-4 text-red-100">
                {result.error}
              </pre>
            )}

            {autodetect && (
              <div className="mb-5 rounded-2xl bg-slate-800 p-4">
                <p className="mb-2 text-sm uppercase tracking-wider text-slate-400">
                  Auto-detected ayah
                </p>

                <div
                  className={`rounded-xl p-4 ${
                    autodetect.accepted
                      ? "bg-emerald-950"
                      : autodetect.needs_confirmation
                      ? "bg-yellow-950"
                      : "bg-red-950"
                  }`}
                >
                  <p className="font-semibold">
                    {autodetect.verdict.replaceAll("_", " ")}
                  </p>
                  <p className="text-sm text-slate-300">
                    Confidence: {(autodetect.confidence * 100).toFixed(2)}%
                  </p>
                </div>

                {result.reference && (
                  <div className="mt-4">
                    <p className="text-sm text-slate-400">
                      Surah {result.reference.surah}, Ayah {result.reference.ayah}
                    </p>
                    <p dir="rtl" className="mt-1 text-2xl leading-loose">
                      {result.reference.text}
                    </p>
                  </div>
                )}
              </div>
            )}

            {gate && (
              <div className="space-y-4">
                <div
                  className={`rounded-2xl p-4 ${
                    gate.accepted ? "bg-emerald-950" : "bg-red-950"
                  }`}
                >
                  <p className="text-lg font-semibold">
                    {gate.accepted ? "Accepted" : "Rejected"} —{" "}
                    {gate.verdict.replaceAll("_", " ")}
                  </p>
                  <p className="text-sm text-slate-300">
                    CER: {(gate.cer * 100).toFixed(2)}% | Character accuracy:{" "}
                    {(gate.char_accuracy * 100).toFixed(2)}% | Edit distance:{" "}
                    {gate.edit_distance}
                  </p>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl bg-slate-800 p-4">
                    <p className="mb-2 text-sm text-slate-400">Expected</p>
                    <p dir="rtl" className="text-2xl leading-loose">
                      {gate.gold}
                    </p>
                  </div>

                  <div className="rounded-2xl bg-slate-800 p-4">
                    <p className="mb-2 text-sm text-slate-400">Recognized</p>
                    <p dir="rtl" className="text-2xl leading-loose">
                      {gate.pred}
                    </p>
                  </div>
                </div>

                {result.message && (
                  <p className="rounded-xl bg-slate-800 p-4 text-slate-300">
                    {result.message}
                  </p>
                )}
              </div>
            )}

            {autodetect?.matches && autodetect.matches.length > 1 && (
              <div className="mt-6">
                <h3 className="mb-3 text-lg font-semibold">
                  Similar ayah matches
                </h3>

                <div className="space-y-3">
                  {autodetect.matches.slice(0, 5).map((match, index) => (
                    <div
                      key={`${match.surah}-${match.ayah}-${index}`}
                      className="rounded-2xl bg-slate-800 p-4"
                    >
                      <div className="mb-1 flex justify-between gap-4 text-sm text-slate-400">
                        <span>
                          Surah {match.surah}, Ayah {match.ayah}
                        </span>
                        <span>CER {(match.cer * 100).toFixed(2)}%</span>
                      </div>
                      <p dir="rtl" className="text-xl leading-loose">
                        {match.text}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  );
}