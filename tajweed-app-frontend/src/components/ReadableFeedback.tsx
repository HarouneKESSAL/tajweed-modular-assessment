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

  // Technical backend fields. Keep them available, but do not show raw
  // "position 41" to the learner.
  position?: number;
  location?: string;
  snippet?: string;
  confidence?: number;
  source_module?: string;
  error_type?: string;

  // Learner-friendly location fields returned by the backend.
  target_word?: string;
  target_letter?: string;
  word_start?: number;
  word_end?: number;
  compact_word_start?: number;
  compact_word_end?: number;

  // Learner-friendly messages returned by the backend.
  location_en?: string;
  location_ar?: string;
  learner_title?: string;
  learner_title_ar?: string;
  learner_message?: string;
  learner_message_ar?: string;

  message: string;
  message_ar?: string;
  corrective_message?: LocalizedText;
};

function getRuleName(item: FeedbackItem): string {
  return (
    item.rule_name_en ||
    item.display_name?.en ||
    item.rule_id ||
    item.rule ||
    "Tajweed rule"
  );
}

function getArabicRuleName(item: FeedbackItem): string {
  return (
    item.rule_name_ar ||
    item.display_name?.ar ||
    item.rule_id ||
    item.rule ||
    "حكم تجويدي"
  );
}

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
            item.feedback_type === "positive" || severity === "positive";

          const ruleName = getRuleName(item);
          const arabicRuleName = getArabicRuleName(item);

          const targetWord = item.target_word || "";
          const targetLetter = item.target_letter || "";

          const title = isPositive
            ? "Well done"
            : item.learner_title || `${ruleName} needs attention`;

          const titleAr = isPositive
            ? "أحسنت"
            : item.learner_title_ar || `${arabicRuleName} يحتاج إلى مراجعة`;

          const message = item.learner_message || item.message;
          const messageAr = item.learner_message_ar || item.message_ar;

          const shouldShowCorrective =
            !item.learner_message &&
            Boolean(item.corrective_message?.en || item.corrective_message?.ar);

          return (
            <li
              key={index}
              className={
                isPositive
                  ? "rounded-xl border border-emerald-700/30 bg-emerald-950/40 p-3 text-sm text-emerald-50"
                  : "rounded-xl border border-amber-700/30 bg-amber-950/30 p-3 text-sm text-amber-100"
              }
            >
              <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
                <div>
                  <div className="font-semibold">{title}</div>

                  {!isPositive && targetWord && (
                    <div className="mt-1 text-sm text-amber-100/90">
                      Word to correct:{" "}
                      <span className="font-bold" dir="rtl">
                        {targetWord}
                      </span>
                      {targetLetter && (
                        <>
                          {" "}
                          <span className="text-amber-100/70">• letter:</span>{" "}
                          <span className="font-bold" dir="rtl">
                            {targetLetter}
                          </span>
                        </>
                      )}
                    </div>
                  )}

                  {!isPositive && !targetWord && item.location && (
                    <div className="mt-1 text-sm text-amber-100/80">
                      Location:{" "}
                      <span dir="rtl" className="font-semibold">
                        {item.location}
                      </span>
                    </div>
                  )}
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

              <div className="leading-relaxed">{message}</div>

              {messageAr && (
                <div dir="rtl" className="mt-2 text-right leading-loose">
                  {messageAr}
                </div>
              )}

              {!isPositive && targetWord && (
                <div dir="rtl" className="mt-2 text-right text-xs text-amber-100/80">
                  {titleAr}
                </div>
              )}

              {shouldShowCorrective && item.corrective_message?.en && (
                <div className="mt-2 text-xs opacity-90">
                  {item.corrective_message.en}
                </div>
              )}

              {shouldShowCorrective && item.corrective_message?.ar && (
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

              {/* Developer/debug details are hidden by default.
                  We keep the raw position available, but do not expose it as the
                  main learner-facing location. */}
              {!isPositive && typeof item.position === "number" && item.position >= 0 && (
                <details className="mt-2 text-xs opacity-70">
                  <summary className="cursor-pointer">Technical details</summary>
                  <div className="mt-1">
                    Internal position: {item.position}
                    {typeof item.word_start === "number" &&
                    typeof item.word_end === "number"
                      ? ` · word span: ${item.word_start}-${item.word_end}`
                      : ""}
                  </div>
                </details>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
