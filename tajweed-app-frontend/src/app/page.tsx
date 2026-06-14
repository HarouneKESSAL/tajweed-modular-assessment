"use client";

import ContentFeedback from "@/components/ContentFeedback";
import MushafPreviewCard from "@/components/MushafPreviewCard";
import ReadableFeedback from "@/components/ReadableFeedback";
import ReferenceAudioPlayer from "@/components/ReferenceAudioPlayer";
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

type RangeMatch = {
  surah?: number | null;
  ayah_start?: number | null;
  ayah_end?: number | null;
  ayah_count?: number;
  text?: string;
  content_text?: string;
  cer?: number;
  avg_cer?: number;
  char_similarity?: number;
  avg_char_similarity?: number;
  edit_distance?: number;
  pred_text?: string;
  verdict?: string;
  confidence?: number;
};

type AutodetectDecision = {
  accepted?: boolean;
  needs_confirmation?: boolean;
  verdict?: string;
  confidence?: number;
};

type AutodetectPayload = {
  accepted?: boolean;
  needs_confirmation?: boolean;
  verdict?: string;
  confidence?: number;
  pred?: string;
  recognized_text?: string;
  strategy?: string;
  best_match?: Match;
  matches?: Match[];
  best_range?: RangeMatch | null;
  decision?: AutodetectDecision;
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

type ContentGatePayload = {
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
  segment_tolerance_applied?: boolean;
};

type ApiResult = {
  ok: boolean;
  request_id?: string;
  mode?: "guided" | "autodetect" | "guided_multi" | "autodetect_multi";
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
  reference_audio?: {
    available: boolean;
    url: string;
    reciter: string;
    surah: number;
    ayah: number;
    format: string;
  } | null;
  autodetect?: AutodetectPayload;
  detected_surah?: number;
  detected_ayah?: number;
  detected_ayah_start?: number;
  detected_ayah_end?: number;
  ayah_start?: number;
  ayah_end?: number;
  expected_segments?: number;
  detected_segments?: number;
  segments?: SegmentPayload[];
  ayah_results?: AyahResult[];
  aggregate?: MultiAyahAggregate;
  content_gate?: ContentGatePayload | null;
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
  reference_audio?: {
  available: boolean;
  url: string;
  reciter: string;
  surah: number;
  ayah: number;
  format: string;
} | null;
  mushaf?: MushafPayload | null;
  content_gate?: ContentGatePayload | null;
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

function formatPercent(value?: number | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}

function formatScore(value?: number | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "N/A";
  return Number(value).toFixed(2);
}

function formatAutodetectVerdict(verdict?: string | null): string {
  const labels: Record<string, string> = {
    contiguous_alignment_accepted: "Auto-detected range accepted",
    contiguous_alignment_needs_confirmation: "Detected range needs confirmation",
    contiguous_alignment_rejected: "Could not confirm detected range",
    segment_range_accepted: "Segmented range accepted",
    segment_range_needs_confirmation: "Segmented range needs confirmation",
    segment_range_not_contiguous: "Segmented ayahs were not contiguous",
    autodetect_range_accepted: "Auto-detected range accepted",
    autodetect_range_needs_confirmation: "Auto-detected range needs confirmation",
    autodetect_range_rejected_low_confidence: "Low-confidence auto-detection",
    accepted_exact: "Accepted — exact match",
    accepted_segment_tolerant: "Accepted — minor ASR variation",
    rejected_content_mismatch: "Content mismatch",
  };

  if (!verdict) return "Auto-detection result";
  return labels[verdict] ?? verdict.replaceAll("_", " ");
}

function getAutodetectMessage(verdict?: string | null): string {
  if (!verdict) return "";

  if (verdict.includes("accepted")) {
    return "The system detected the recited range and started assessment automatically.";
  }

  if (verdict.includes("needs_confirmation")) {
    return "The system found a likely range, but confidence was not high enough for automatic Tajweed scoring.";
  }

  if (verdict.includes("not_contiguous")) {
    return "The detected ayahs were not continuous. Try again with clearer pauses between ayahs.";
  }

  return "The system could not confidently detect the recited range.";
}

function getAutodetectState(autodetect?: AutodetectPayload) {
  const decision = autodetect?.decision;
  const accepted = Boolean(autodetect?.accepted ?? decision?.accepted);
  const needsConfirmation = Boolean(
    autodetect?.needs_confirmation ?? decision?.needs_confirmation
  );
  const verdict = autodetect?.verdict ?? decision?.verdict ?? null;
  const confidence = autodetect?.confidence ?? decision?.confidence ?? null;

  return { accepted, needsConfirmation, verdict, confidence };
}

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
      const response = await fetch("https://upset-mails-cross.loca.lt/api/assess-recitation", {
        method: "POST",
        body: form,
        cache: "no-store",
        headers: {
          "Cache-Control": "no-cache",
        },
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ ok: false, error: String(error) });
    } finally {
      setLoading(false);
    }
  }

  const gate = result?.content_gate;
  const autodetect = result?.autodetect;
  const autodetectState = getAutodetectState(autodetect);
  const isMultiAyah = result?.mode === "guided_multi" || result?.mode === "autodetect_multi";
  const ayahResults = result?.ayah_results ?? [];
  const aggregate = result?.aggregate;
  const mushaf = result?.mushaf;
  const tajweedUi = result?.tajweed_ui;
  const contentFeedback = result?.content_feedback;
  const tajweed = result?.tajweed;
  const tajweedScore =
    tajweed?.result?.weighted_score ??
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
                Record without selecting an ayah. For multi-ayah recitation, pause
                briefly between ayahs.
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
            <div className="mb-5 grid grid-cols-1 gap-4 md:grid-cols-3">
              <label className="space-y-2">
                <span className="text-sm text-slate-300">Surah number</span>
                <input
                  className="w-full rounded-xl bg-slate-800 px-3 py-2 outline-none ring-1 ring-slate-700 focus:ring-blue-400"
                  value={surah}
                  onChange={(event) => setSurah(event.target.value)}
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm text-slate-300">Ayah number</span>
                <input
                  className="w-full rounded-xl bg-slate-800 px-3 py-2 outline-none ring-1 ring-slate-700 focus:ring-blue-400"
                  value={ayah}
                  onChange={(event) => setAyah(event.target.value)}
                />
              </label>

              <label className="space-y-2">
                <span className="text-sm text-slate-300">To ayah number</span>
                <input
                  value={ayahEnd}
                  onChange={(event) => setAyahEnd(event.target.value)}
                  placeholder="optional"
                  className="w-full rounded-xl bg-slate-800 px-3 py-2 text-white outline-none ring-1 ring-slate-700 focus:ring-blue-400"
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
                  Auto-detected recitation
                </p>

                <div
                  className={`rounded-xl p-4 ${
                    autodetectState.accepted
                      ? "bg-emerald-950"
                      : autodetectState.needsConfirmation
                      ? "bg-yellow-950"
                      : "bg-red-950"
                  }`}
                >
                  <p className="font-semibold">
                    {formatAutodetectVerdict(autodetectState.verdict)}
                  </p>
                  <p className="text-sm text-slate-300">
                    Confidence: {formatPercent(autodetectState.confidence)}
                  </p>
                  {getAutodetectMessage(autodetectState.verdict) && (
                    <p className="mt-1 text-xs text-slate-300">
                      {getAutodetectMessage(autodetectState.verdict)}
                    </p>
                  )}
                </div>

                {(autodetect.recognized_text || autodetect.pred) && (
                  <div className="mt-4 rounded-xl bg-slate-900 p-3">
                    <p className="mb-1 text-xs uppercase tracking-wider text-slate-500">
                      Recognized
                    </p>
                    <p dir="rtl" className="text-lg leading-loose text-white">
                      {autodetect.recognized_text ?? autodetect.pred}
                    </p>
                  </div>
                )}

                {autodetect.best_range && (
                  <div className="mt-4">
                    <p className="text-sm text-slate-400">
                      Detected range: Surah {autodetect.best_range.surah ?? "?"}, Ayah{" "}
                      {autodetect.best_range.ayah_start ?? "?"} to{" "}
                      {autodetect.best_range.ayah_end ?? "?"}
                    </p>
                    {(autodetect.best_range.text || autodetect.best_range.content_text) && (
                      <p dir="rtl" className="mt-1 text-xl leading-loose">
                        {autodetect.best_range.text ?? autodetect.best_range.content_text}
                      </p>
                    )}
                  </div>
                )}

                {result.reference && !isMultiAyah && (
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
                      {result.mode === "autodetect_multi"
                        ? "Free recitation multi-ayah assessment"
                        : "Multi-ayah guided assessment"}
                    </h3>

                    <p className="mt-1 text-sm text-slate-400">
                      Surah {result.surah}, Ayah {result.ayah_start} to {result.ayah_end}
                    </p>

                    {result.segmentation_strategy && (
                      <p className="mt-1 text-xs text-slate-500">
                        Segmentation: {result.segmentation_strategy.replaceAll("_", " ")}
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
                      {formatScore(aggregate?.average_tajweed_score)}
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
                    const hasContentFeedback = Boolean(item.content_feedback?.items?.length);

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
                            <h4 className="font-semibold text-white">Ayah {item.ayah}</h4>
                            <p className="text-xs text-slate-400">
                              Segment {item.segment?.index} •{" "}
                              {item.segment?.start_sec?.toFixed?.(2)}s →{" "}
                              {item.segment?.end_sec?.toFixed?.(2)}s
                            </p>
                          </div>

                          <div className="flex flex-wrap gap-2">
                            <span className={`rounded-full px-3 py-1 text-xs font-semibold ${statusClass}`}>
                              {statusLabel}
                            </span>

                            <span className="rounded-full bg-blue-500/20 px-3 py-1 text-xs font-semibold text-blue-200">
                              Score {formatScore(score)}
                            </span>
                          </div>
                        </div>

                        {item.reference?.text && (
                          <p dir="rtl" className="mt-3 text-right text-xl leading-loose text-white">
                            {item.reference.text}
                          </p>
                        )}
                        <ReferenceAudioPlayer audio={item.reference_audio} />
                        {item.mushaf && (
                          <details className="mt-3 rounded-xl border border-slate-700 bg-slate-900/60 p-3">
                            <summary className="cursor-pointer text-sm font-semibold text-slate-200">
                              Show reference Mushaf Moulawan preview
                            </summary>

                            <div className="mt-3">
                              <MushafPreviewCard mushaf={item.mushaf} />
                            </div>
                          </details>
                        )}

                        {item.content_gate && (
                          <div className="mt-3 rounded-xl bg-slate-900/80 p-3 text-sm text-slate-300">
                            <p className="truncate">
                              <span className="font-semibold text-slate-200">Recognized:</span>{" "}
                              <span dir="rtl">{item.content_gate.pred}</span>
                            </p>
                            <p className="mt-1 text-xs text-slate-400">
                              CER: {formatPercent(item.content_gate.cer)} | Character accuracy:{" "}
                              {formatPercent(item.content_gate.char_accuracy)}
                              {item.content_gate.segment_tolerance_applied && (
                                <span className="ml-2 text-emerald-300">
                                  Accepted with segment tolerance
                                </span>
                              )}
                            </p>
                          </div>
                        )}

                        {!accepted && item.content_feedback && (
                          <div className="mt-3 rounded-xl border border-red-900/60 bg-red-950/40 p-3">
                            <p className="text-sm font-semibold text-red-100">
                              Content needs correction before Tajweed scoring.
                            </p>
                            {item.content_feedback.items?.slice(0, 2).map((feedback, index) => (
                              <p key={index} className="mt-2 text-xs text-red-100/90">
                                Expected <span className="font-semibold">“{feedback.expected}”</span>, but
                                recognized <span className="font-semibold">“{feedback.recognized}”</span>.
                              </p>
                            ))}
                          </div>
                        )}

                        {accepted && hasTajweedErrors && feedbackItems.length > 0 && (
                          <div className="mt-3 rounded-xl border border-amber-900/60 bg-amber-950/30 p-3">
                            <p className="text-sm font-semibold text-amber-100">
                              {item.tajweed_score?.num_errors} Tajweed issue
                              {item.tajweed_score?.num_errors === 1 ? "" : "s"} detected.
                            </p>
                            <p className="mt-1 text-xs text-amber-100/80">
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
                    {gate.accepted ? "Accepted" : "Rejected"} — {formatAutodetectVerdict(gate.verdict)}
                  </p>
                  <p className="text-sm text-slate-300">
                    CER: {formatPercent(gate.cer)} | Character accuracy: {formatPercent(gate.char_accuracy)} | Edit distance:{" "}
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
                <ReferenceAudioPlayer audio={result.reference_audio} />
                <SupportedRules items={tajweedUi?.supported_rules} />
                <ReadableFeedback items={tajweedUi?.readable_feedback} />

                {tajweed && (
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
                            Score: {formatScore(tajweedScore?.score)}/100
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
                              {tajweedDiagnosis.feedback.map((item, index) => (
                                <li key={index}>{item}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {moduleJudgments.length > 0 && (
                          <div className="rounded-xl bg-slate-900 p-4">
                            <p className="mb-3 text-sm text-slate-400">Rule-level findings</p>

                            <div className="space-y-3">
                              {moduleJudgments.slice(0, 10).map((judgment, index) => (
                                <div
                                  key={index}
                                  className={`rounded-xl p-3 ${
                                    judgment.is_correct ? "bg-emerald-950" : "bg-red-950"
                                  }`}
                                >
                                  <div className="flex flex-wrap justify-between gap-2">
                                    <p className="font-semibold">
                                      {judgment.rule} at position {judgment.position}
                                    </p>
                                    <p className="text-sm">
                                      {judgment.is_correct ? "Correct" : "Needs attention"}
                                    </p>
                                  </div>

                                  <p className="mt-1 text-sm text-slate-300">
                                    Predicted: {judgment.predicted_rule ?? "N/A"} | Source:{" "}
                                    {judgment.source_module ?? "N/A"} | Confidence:{" "}
                                    {judgment.confidence !== undefined
                                      ? Number(judgment.confidence).toFixed(3)
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

            {autodetect?.matches && autodetect.matches.length > 1 && !autodetectState.accepted && (
              <div className="mt-6">
                <h3 className="mb-3 text-lg font-semibold">Similar ayah matches</h3>

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
                        <span>CER {formatPercent(match.cer)}</span>
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
