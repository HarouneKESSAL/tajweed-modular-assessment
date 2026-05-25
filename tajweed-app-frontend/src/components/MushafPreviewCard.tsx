type MushafSegment = {
  text: string;
  rule?: string | null;
  color?: string | null;
};

type MushafPayload = {
  available?: boolean;
  surah?: number;
  ayah?: number;
  text?: string;
  segments?: MushafSegment[];
  reason?: string;
};

const fallbackColor = "#111827";

const ruleLabels: Record<string, string> = {
  madd: "Madd",
  ghunnah: "Ghunnah",
  ikhfa: "Ikhfa",
  idgham: "Idgham",
  qalqalah: "Qalqalah",
  iqlab: "Iqlab",
};

const fallbackLegend = [
  { rule: "madd", label: "Madd", color: "#FF6666" },
  { rule: "ikhfa", label: "Ikhfa", color: "#00AA00" },
  { rule: "idgham", label: "Idgham", color: "#7c3aed" },
  { rule: "qalqalah", label: "Qalqalah", color: "#0066FF" },
];

export default function MushafPreviewCard({
  mushaf,
}: {
  mushaf?: MushafPayload | null;
}) {
  if (!mushaf) return null;

  if (!mushaf.available) {
    return (
      <div className="rounded-2xl bg-slate-800 p-4">
        <h3 className="mb-2 text-lg font-semibold">
          Reference Mushaf Moulawan preview
        </h3>
        <div className="rounded-xl bg-amber-950 p-4 text-amber-100">
          <p className="font-semibold">Colored preview is not available yet.</p>
          {mushaf.reason && <p className="mt-1 text-sm">{mushaf.reason}</p>}
        </div>
      </div>
    );
  }

  const segments = mushaf.segments || [];
  const legend = buildLegend(segments);

  return (
    <div className="rounded-2xl bg-slate-800 p-4">
      <h3 className="mb-3 text-lg font-semibold">
        Reference Mushaf Moulawan preview
      </h3>

      <div className="rounded-2xl border border-amber-200/40 bg-[#f8f2e8] p-6 shadow-inner">
        <div className="mb-3 flex items-center justify-between gap-3 text-slate-700">
          <div className="text-xs font-semibold uppercase tracking-[0.25em]">
            Mushaf Moulawan
          </div>

          <div className="text-sm">
            Surah {mushaf.surah}, Ayah {mushaf.ayah}
          </div>
        </div>

        <div
          dir="rtl"
          className="text-right text-3xl leading-loose md:text-4xl"
          style={{
            fontFamily:
              '"Amiri Quran", "Scheherazade New", "Traditional Arabic", serif',
            lineHeight: 2.1,
          }}
        >
          {segments.length > 0 ? (
            segments.map((seg, index) => (
              <span
                key={`${index}-${seg.text}`}
                title={seg.rule || ""}
                className="whitespace-pre-wrap"
                style={{
                  color: seg.color || fallbackColor,
                  fontWeight: seg.rule ? 700 : 500,
                }}
              >
                {seg.text}
              </span>
            ))
          ) : (
            <span className="text-slate-900">{mushaf.text}</span>
          )}
        </div>

        <div className="mt-5 grid gap-2 text-xs text-slate-700 sm:grid-cols-2 md:grid-cols-3">
          {legend.map((item) => (
            <LegendItem
              key={`${item.rule}-${item.color}`}
              color={item.color}
              label={item.label}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function buildLegend(segments: MushafSegment[]) {
  const seen = new Map<string, { rule: string; label: string; color: string }>();

  for (const seg of segments) {
    if (!seg.rule || !seg.color) continue;

    const rule = String(seg.rule);
    if (seen.has(rule)) continue;

    seen.set(rule, {
      rule,
      label: ruleLabels[rule] || rule,
      color: seg.color,
    });
  }

  const dynamicLegend = Array.from(seen.values());

  if (dynamicLegend.length > 0) {
    return dynamicLegend;
  }

  return fallbackLegend;
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="inline-block h-3 w-3 rounded-full"
        style={{ backgroundColor: color }}
      />
      <span>{label}</span>
    </div>
  );
}
