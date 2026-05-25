type Localized = {
  en?: string;
  ar?: string;
};

type ContentFeedbackItem = {
  feedback_type?: string;
  error_type?: string;
  title?: Localized;
  expected?: string;
  recognized?: string;
  severity_level?: string;
  severity_score?: number;
  message?: Localized;
  default_error_message?: Localized;
  corrective_message?: Localized;
  position?: {
    expected_word_index?: number;
    recognized_word_index?: number;
  };
};

type ContentFeedbackPayload = {
  available?: boolean;
  accepted?: boolean;
  summary?: Localized;
  expected?: string;
  recognized?: string;
  metrics?: {
    char_accuracy?: number;
    cer?: number;
    edit_distance?: number;
  };
  items?: ContentFeedbackItem[];
  reason?: string;
};

export default function ContentFeedback({
  feedback,
}: {
  feedback?: ContentFeedbackPayload | null;
}) {
  if (!feedback) return null;

  if (feedback.accepted) return null;

  const items = feedback.items || [];

  return (
    <div className="rounded-2xl bg-slate-800 p-4">
      <h3 className="mb-3 text-lg font-semibold text-red-100">
        Content feedback
      </h3>

      <div className="mb-4 rounded-xl border border-red-700/40 bg-red-950/40 p-4 text-red-100">
        <p className="font-semibold">
          {feedback.summary?.en || "Content needs correction before Tajweed scoring."}
        </p>

        {feedback.summary?.ar && (
          <p dir="rtl" className="mt-2 text-right leading-loose">
            {feedback.summary.ar}
          </p>
        )}

        {feedback.reason && (
          <p className="mt-2 text-sm text-red-200">{feedback.reason}</p>
        )}

        {feedback.metrics && (
          <p className="mt-3 text-xs text-red-200/90">
            CER:{" "}
            {typeof feedback.metrics.cer === "number"
              ? `${(feedback.metrics.cer * 100).toFixed(2)}%`
              : "N/A"}
            {" | "}
            Character accuracy:{" "}
            {typeof feedback.metrics.char_accuracy === "number"
              ? `${(feedback.metrics.char_accuracy * 100).toFixed(2)}%`
              : "N/A"}
            {" | "}
            Edit distance: {feedback.metrics.edit_distance ?? "N/A"}
          </p>
        )}
      </div>

      <div className="mb-4 grid gap-3 md:grid-cols-2">
        <div className="rounded-xl bg-slate-900 p-3">
          <p className="mb-2 text-sm text-slate-400">Expected</p>
          <p dir="rtl" className="text-right text-xl leading-loose">
            {feedback.expected || "—"}
          </p>
        </div>

        <div className="rounded-xl bg-slate-900 p-3">
          <p className="mb-2 text-sm text-slate-400">Recognized</p>
          <p dir="rtl" className="text-right text-xl leading-loose">
            {feedback.recognized || "—"}
          </p>
        </div>
      </div>

      {items.length > 0 && (
        <ul className="space-y-3">
          {items.map((item, index) => (
            <li
              key={index}
              className="rounded-xl border border-red-800/40 bg-red-950/30 p-3 text-sm text-red-50"
            >
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div className="font-semibold">
                  {item.title?.en || item.error_type || "Content issue"}
                </div>

                <div className="flex flex-wrap gap-2 text-xs">
                  {item.severity_level && (
                    <span className="rounded-full bg-slate-900 px-2 py-1">
                      {item.severity_level}
                    </span>
                  )}

                  {typeof item.severity_score === "number" && (
                    <span className="rounded-full bg-slate-900 px-2 py-1">
                      priority {item.severity_score}
                    </span>
                  )}

                  {item.error_type && (
                    <span className="rounded-full bg-slate-900 px-2 py-1">
                      {item.error_type}
                    </span>
                  )}
                </div>
              </div>

              {item.title?.ar && (
                <div dir="rtl" className="mb-2 text-right font-semibold">
                  {item.title.ar}
                </div>
              )}

              <div className="mb-3 grid gap-2 md:grid-cols-2">
                {item.expected && (
                  <div className="rounded-lg bg-slate-950/70 p-2">
                    <p className="text-xs text-slate-400">Expected part</p>
                    <p dir="rtl" className="mt-1 text-right text-lg">
                      {item.expected}
                    </p>
                  </div>
                )}

                {item.recognized && (
                  <div className="rounded-lg bg-slate-950/70 p-2">
                    <p className="text-xs text-slate-400">Recognized part</p>
                    <p dir="rtl" className="mt-1 text-right text-lg">
                      {item.recognized}
                    </p>
                  </div>
                )}
              </div>

              {item.message?.en && (
                <p className="leading-relaxed">{item.message.en}</p>
              )}

              {item.message?.ar && (
                <p dir="rtl" className="mt-2 text-right leading-loose">
                  {item.message.ar}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
