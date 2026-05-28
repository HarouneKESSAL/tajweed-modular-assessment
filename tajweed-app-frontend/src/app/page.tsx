"use client";

import ContentFeedback from "@/components/ContentFeedback";

import MushafPreviewCard from "@/components/MushafPreviewCard";
import ReadableFeedback from "@/components/ReadableFeedback";
import SupportedRules from "@/components/SupportedRules";

import { useRef, useState } from "react";

type Match = {
  surah: number;
  ayah: number;
  text: string;
  cer: number;
  char_similarity: number;
  edit_distance: number;
};

type LocalizedText = {
  en?: string;
  ar?: string;
};

type MushafPayload = {
  available?: boolean;
  surah?: number;
  ayah?: number;
  text?: string;
  segments?: {
    text: string;
    rule?: string | null;
    color?: string | null;
  }[];
  reason?: string;
};

type ContentFeedbackPayload = {
  available?: boolean;
  accepted?: boolean;
  summary?: LocalizedText;
  expected?: string;
  recognized?: string;
  metrics?: {
    char_accuracy?: number;
    cer?: number;
    edit_distance?: number;
  };
  items?: {
    feedback_type?: string;
    error_type?: string;
    title?: LocalizedText;
    expected?: string;
    recognized?: string;
    severity_level?: string;
    severity_score?: number;
    message?: LocalizedText;
    default_error_message?: LocalizedText;
    corrective_message?: LocalizedText;
    position?: {
      expected_word_index?: number;
      recognized_word_index?: number;
    };
  }[];
  reason?: string;
};

type SupportedRule = {
  name: string;
  module: string;
  status?: string;
};

type ReadableFeedbackItem = {
  feedback_type?: string;
  severity?: string;
  severity_level?: string;
  severity_score?: number;
  rule?: string;
  rule_id?: string;
  rule_name_en?: string;
  rule_name_ar?: string;
  display_name?: LocalizedText;
  position?: number;
  location?: string;
  confidence?: number;
  source_module?: string;
  error_type?: string;
  message: string;
  message_ar?: string;
  corrective_message?: LocalizedText;
};

type TajweedScore = {
  score?: number;
  num_errors?: number;
  weighted_error_sum?: number;
};

type ModuleJudgment = {
  is_correct?: boolean;
  rule?: string;
  position?: number;
  predicted_rule?: string;
  source_module?: string;
  confidence?: number;
};

type TajweedDiagnosis = {
  weighted_score?: TajweedScore;
  feedback?: string[];
  module_judgments?: ModuleJudgment[];
};

type TajweedPayload = {
  available?: boolean;
  reason?: string;
  matched_row?: {
    sample_id?: string;
    sample_index?: number;
    text?: string;
  };
  result?: {
    weighted_score?: TajweedScore;
    diagnosis?: TajweedDiagnosis;
  };
};

type ApiResult = {
  ok: boolean;
  request_id?: string;
  mode?: "guided" | "autodetect" | "guided_multi";
  segmentation_strategy?: string;
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
  ayah_start?: number;
  ayah_end?: number;
  expected_segments?: number;
  detected_segments?: number;
  segments?: SegmentPayload[];
  ayah_results?: AyahResult[];
  aggregate?: MultiAyahAggregate;
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
  mushaf?: MushafPayload | null;
  tajweed_ui?: {
    supported_rules?: SupportedRule[];
    readable_feedback?: ReadableFeedbackItem[];
  } | null;
  content_feedback?: ContentFeedbackPayload | null;
  tajweed?: TajweedPayload | null;
  message?: string;
  error?: string;
};


type SegmentPayload = {
  index: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  audio_path?: string;
  method?: string;
};

type AyahResult = {
  surah: number;
  ayah: number;
  segment?: SegmentPayload;
  reference?: {
    surah: number;
    ayah: number;
    text: string;
    text_compact?: string;
    source_id?: string;
  };
  content_gate?: ApiResult["content_gate"];
  content_feedback?: ContentFeedbackPayload | null;
  tajweed?: TajweedPayload | null;
  tajweed_ui?: {
    supported_rules?: SupportedRule[];
    readable_feedback?: ReadableFeedbackItem[];
  } | null;
  tajweed_score?: TajweedScore | null;
  message?: string;
};

type MultiAyahAggregate = {
  content_accepted_count?: number;
  content_total?: number;
  content_acceptance_rate?: number;
  tajweed_available_count?: number;
  average_tajweed_score?: number | null;
  total_errors?: number;
};
export default function Home() {
  const [mode, setMode] = useState<"guided" | "autodetect">("autodetect");
  const [surah, setSurah] = useState("1");
  const [ayah, setAyah] = useState("1");
  const [ayahEnd, setAyahEnd] = useState("");

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

      if (ayahEnd.trim()) {
        form.append("ayah_end", ayahEnd.trim());
      }
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
const isMultiAyah = result?.mode === "guided_multi";
const ayahResults = result?.ayah_results ?? [];
const aggregate = result?.aggregate;
const mushaf = result?.mushaf;
const tajweedUi = result?.tajweed_ui;
const contentFeedback = result?.content_feedback;
const tajweed = result?.tajweed;
const tajweedScore =
  tajweed?.result?.weighted_score ??
  tajweed?.result?.diagnosis?.weighted_score ??
  tajweed?.result?.diagnosis?.weighted_score ??
  null;
const tajweedDiagnosis = tajweed?.result?.diagnosis ?? null;
const moduleJudgments: ModuleJudgment[] = tajweedDiagnosis?.module_judgments ?? [];

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
            <div className="mb-5 grid grid-cols-1 md:grid-cols-3 gap-4">
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

              <label className="text-xs text-slate-300">
                To ayah number
                <input
                  value={ayahEnd}
                  onChange={(e) => setAyahEnd(e.target.value)}
                  placeholder="optional"
                  className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-white"
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

            {isMultiAyah && (
              <div className="mt-4 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h3 className="text-base font-semibold text-white">
                      Multi-ayah guided assessment
                    </h3>

                    <p className="mt-1 text-sm text-slate-400">
                      Surah {result.surah}, Ayah {result.ayah_start} to{" "}
                      {result.ayah_end}
                    </p>

                    {result.segmentation_strategy && (
                      <p className="mt-1 text-xs text-slate-500">
                        Segmentation:{" "}
                        {result.segmentation_strategy.replaceAll("_", " ")}
                      </p>
                    )}
                  </div>
                </div>

                <div className="mt-4 grid gap-3 md:grid-cols-4">
                  <div className="rounded-xl bg-slate-800 p-3">
                    <p className="text-xs text-slate-400">Segments</p>
                    <p className="text-xl font-bold text-white">
                      {result.detected_segments}/{result.expected_segments}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-800 p-3">
                    <p className="text-xs text-slate-400">Content accepted</p>
                    <p className="text-xl font-bold text-white">
                      {aggregate?.content_accepted_count}/{aggregate?.content_total}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-800 p-3">
                    <p className="text-xs text-slate-400">Average Tajweed score</p>
                    <p className="text-xl font-bold text-white">
                      {aggregate?.average_tajweed_score !== null &&
                      aggregate?.average_tajweed_score !== undefined
                        ? Number(aggregate.average_tajweed_score).toFixed(2)
                        : "N/A"}
                    </p>
                  </div>

                  <div className="rounded-xl bg-slate-800 p-3">
                    <p className="text-xs text-slate-400">Total errors</p>
                    <p className="text-xl font-bold text-white">
                      {aggregate?.total_errors ?? 0}
                    </p>
                  </div>
                </div>

                <div className="mt-5 space-y-3">
                  {ayahResults.map((item) => {
                    const accepted = Boolean(item.content_gate?.accepted);
                    const score = item.tajweed_score?.score;
                    const feedbackItems = item.tajweed_ui?.readable_feedback ?? [];
                    const hasTajweedErrors = Boolean(item.tajweed_score?.num_errors);
                    const hasContentFeedback = Boolean(
                      item.content_feedback?.items?.length
                    );

                    const statusLabel = accepted
                      ? hasTajweedErrors
                        ? "Tajweed needs attention"
                        : "Accepted"
                      : "Content rejected";

                    const statusClass = accepted
                      ? hasTajweedErrors
                        ? "bg-amber-500/20 text-amber-200"
                        : "bg-emerald-500/20 text-emerald-200"
                      : "bg-red-500/20 text-red-200";

                    return (
                      <div
                        key={`${item.surah}-${item.ayah}`}
                        className="rounded-2xl border border-slate-700 bg-slate-800/70 p-4"
                      >
                        <div className="flex flex-wrap items-start justify-between gap-3">
                          <div>
                            <h4 className="font-semibold text-white">
                              Ayah {item.ayah}
                            </h4>
                            <p className="text-xs text-slate-400">
                              Segment {item.segment?.index} •{" "}
                              {item.segment?.start_sec?.toFixed?.(2)}s →{" "}
                              {item.segment?.end_sec?.toFixed?.(2)}s
                            </p>
                          </div>

                          <div className="flex flex-wrap gap-2">
                            <span
                              className={`rounded-full px-3 py-1 text-xs font-semibold ${statusClass}`}
                            >
                              {statusLabel}
                            </span>

                            <span className="rounded-full bg-blue-500/20 px-3 py-1 text-xs font-semibold text-blue-200">
                              Score{" "}
                              {score !== undefined && score !== null
                                ? Number(score).toFixed(2)
                                : "N/A"}
                            </span>
                          </div>
                        </div>

                        {item.reference?.text && (
                          <p
                            dir="rtl"
                            className="mt-3 text-right text-xl leading-loose text-white"
                          >
                            {item.reference.text}
                          </p>
                        )}

                        {item.content_gate && (
                          <div className="mt-3 rounded-xl bg-slate-900/80 p-3 text-sm text-slate-300">
                            <p className="truncate">
                              <span className="font-semibold text-slate-200">
                                Recognized:
                              </span>{" "}
                              <span dir="rtl">{item.content_gate.pred}</span>
                            </p>

                            <p className="mt-1 text-xs text-slate-400">
                              CER: {(item.content_gate.cer * 100).toFixed(2)}% |
                              Character accuracy:{" "}
                              {(item.content_gate.char_accuracy * 100).toFixed(2)}%
                            </p>
                          </div>
                        )}

                        {!accepted && item.content_feedback && (
                          <div className="mt-3 rounded-xl border border-red-900/60 bg-red-950/40 p-3">
                            <p className="text-sm font-semibold text-red-100">
                              Content needs correction before Tajweed scoring.
                            </p>

                            {item.content_feedback.items?.slice(0, 2).map((fb, index) => (
                              <p
                                key={`${fb.expected}-${fb.recognized}-${index}`}
                                className="mt-2 text-xs text-red-100/90"
                              >
                                Expected{" "}
                                <span className="font-semibold">“{fb.expected}”</span>,
                                but recognized{" "}
                                <span className="font-semibold">
                                  “{fb.recognized}”
                                </span>
                                .
                              </p>
                            ))}
                          </div>
                        )}

                        {accepted && hasTajweedErrors && feedbackItems.length > 0 && (
                          <div className="mt-3 rounded-xl border border-amber-900/60 bg-amber-950/30 p-3">
                            <p className="text-sm font-semibold text-amber-100">
                              {item.tajweed_score?.num_errors} Tajweed issue
                              {item.tajweed_score?.num_errors === 1 ? "" : "s"}{" "}
                              detected.
                            </p>

                            <p className="mt-1 line-clamp-2 text-xs text-amber-100/80">
                              {feedbackItems[0]?.message}
                            </p>
                          </div>
                        )}

                        {accepted && !hasTajweedErrors && (
                          <div className="mt-3 rounded-xl border border-emerald-900/60 bg-emerald-950/30 p-3">
                            <p className="text-sm font-semibold text-emerald-100">
                              No Tajweed errors detected.
                            </p>
                          </div>
                        )}

                        {(hasContentFeedback || feedbackItems.length > 0) && (
                          <details className="mt-3 rounded-xl border border-slate-700 bg-slate-900/60 p-3">
                            <summary className="cursor-pointer text-sm font-semibold text-slate-200">
                              Show detailed feedback
                            </summary>

                            <div className="mt-3 space-y-3">
                              {!accepted && item.content_feedback && (
                                <ContentFeedback feedback={item.content_feedback} />
                              )}

                              {accepted && feedbackItems.length > 0 && (
                                <ReadableFeedback items={feedbackItems} />
                              )}
                            </div>
                          </details>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

           {!isMultiAyah && gate && (
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

              <ContentFeedback feedback={contentFeedback} />

              <MushafPreviewCard mushaf={mushaf} />

              <SupportedRules items={tajweedUi?.supported_rules} />

              <ReadableFeedback items={tajweedUi?.readable_feedback} />

{!isMultiAyah && tajweed && (
  <div className="rounded-2xl bg-slate-800 p-4">
    <h3 className="mb-3 text-lg font-semibold">Tajweed assessment</h3>

    {!tajweed.available && (
      <div className="rounded-xl bg-yellow-950 p-4 text-yellow-100">
        <p className="font-semibold">Tajweed not available for this ayah yet</p>
        <p className="mt-1 text-sm">{tajweed.reason}</p>
      </div>
    )}

    {tajweed.available && (
      <div className="space-y-4">
        <div className="rounded-xl bg-emerald-950 p-4">
          <p className="text-lg font-semibold">
            Score:{" "}
            {tajweedScore?.score !== undefined
              ? Number(tajweedScore.score).toFixed(2)
              : "N/A"}
            /100
          </p>

          {tajweedScore && (
            <p className="text-sm text-slate-300">
              Errors: {tajweedScore.num_errors ?? 0} | Weighted error sum:{" "}
              {tajweedScore.weighted_error_sum ?? 0}
            </p>
          )}
        </div>

        {tajweed.matched_row && (
          <div className="rounded-xl bg-slate-900 p-4">
            <p className="text-sm text-slate-400">Matched Tajweed row</p>
            <p className="mt-1 text-sm">
              {tajweed.matched_row.sample_id} — sample index{" "}
              {tajweed.matched_row.sample_index}
            </p>
            <p dir="rtl" className="mt-2 text-xl leading-loose">
              {tajweed.matched_row.text}
            </p>
          </div>
        )}

        {tajweedDiagnosis?.feedback && (
          <div className="rounded-xl bg-slate-900 p-4">
            <p className="mb-2 text-sm text-slate-400">Feedback</p>
            <ul className="list-inside list-disc space-y-1 text-slate-200">
              {tajweedDiagnosis.feedback.map((item: string, index: number) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        {moduleJudgments.length > 0 && (
          <div className="rounded-xl bg-slate-900 p-4">
            <p className="mb-3 text-sm text-slate-400">Rule-level findings</p>

            <div className="space-y-3">
              {moduleJudgments.slice(0, 10).map((j, index: number) => (
                <div
                  key={index}
                  className={`rounded-xl p-3 ${
                    j.is_correct ? "bg-emerald-950" : "bg-red-950"
                  }`}
                >
                  <div className="flex flex-wrap justify-between gap-2">
                    <p className="font-semibold">
                      {j.rule} at position {j.position}
                    </p>
                    <p className="text-sm">
                      {j.is_correct ? "Correct" : "Needs attention"}
                    </p>
                  </div>

                  <p className="mt-1 text-sm text-slate-300">
                    Predicted: {j.predicted_rule ?? "N/A"} | Source:{" "}
                    {j.source_module ?? "N/A"} | Confidence:{" "}
                    {j.confidence !== undefined
                      ? Number(j.confidence).toFixed(3)
                      : "N/A"}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )}
  </div>
)}
              </div>
            )}

            {autodetect?.matches &&
              autodetect.matches.length > 1 &&
              !autodetect.accepted && (
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
