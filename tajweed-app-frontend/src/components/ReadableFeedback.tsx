type LocalizedText = {
  en?: string;
  ar?: string;
};

type FeedbackItem = {
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

export default function ReadableFeedback({
  items,
}: {
  items?: FeedbackItem[] | null;
}) {
  if (!items || items.length === 0) return null;

  const hasOnlyPositive = items.every(
    (item) =>
      item.feedback_type === "positive" ||
      item.severity === "positive" ||
      item.severity_level === "positive"
  );

  return (
    <div className="rounded-2xl bg-slate-800 p-4">
      <h3 className="mb-3 text-lg font-semibold">
        {hasOnlyPositive ? "Encouragement" : "Learner feedback"}
      </h3>

      <ul className="space-y-3">
        {items.map((item, index) => {
          const severity = item.severity_level || item.severity || "medium";
          const isPositive =
            item.feedback_type === "positive" ||
            severity === "positive";

          const ruleName =
            item.rule_name_en ||
            item.display_name?.en ||
            item.rule_id ||
            item.rule ||
            "Tajweed rule";

          return (
            <li
              key={index}
              className={
                isPositive
                  ? "rounded-xl border border-emerald-700/30 bg-emerald-950/40 p-3 text-sm text-emerald-50"
                  : "rounded-xl border border-amber-700/30 bg-amber-950/30 p-3 text-sm text-amber-100"
              }
            >
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div className="font-semibold">
                  {isPositive ? "Well done" : ruleName}
                  {!isPositive &&
                  typeof item.position === "number" &&
                  item.position >= 0
                    ? ` · position ${item.position}`
                    : ""}
                </div>

                <div className="flex flex-wrap gap-2 text-xs">
                  <span className="rounded-full bg-slate-900 px-2 py-1 text-slate-200">
                    {isPositive ? "positive" : severity}
                  </span>

                  {!isPositive && typeof item.severity_score === "number" && (
                    <span className="rounded-full bg-slate-900 px-2 py-1 text-slate-200">
                      priority {item.severity_score}
                    </span>
                  )}

                  {!isPositive && item.source_module && (
                    <span className="rounded-full bg-slate-900 px-2 py-1 text-slate-200">
                      {item.source_module}
                    </span>
                  )}
                </div>
              </div>

              <div className="leading-relaxed">{item.message}</div>

              {item.message_ar && (
                <div dir="rtl" className="mt-2 text-right leading-loose">
                  {item.message_ar}
                </div>
              )}

              {item.corrective_message?.en && (
                <div className="mt-2 text-xs opacity-90">
                  {item.corrective_message.en}
                </div>
              )}

              {item.corrective_message?.ar && (
                <div dir="rtl" className="mt-1 text-right text-xs opacity-90">
                  {item.corrective_message.ar}
                </div>
              )}

              {!isPositive && item.error_type && (
                <div className="mt-2 text-xs opacity-80">
                  Error type: {item.error_type}
                  {typeof item.confidence === "number"
                    ? ` · Confidence: ${(item.confidence * 100).toFixed(1)}%`
                    : ""}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
