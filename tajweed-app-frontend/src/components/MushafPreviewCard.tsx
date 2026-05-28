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

const ruleColors: Record<string, string> = {
  madd: "#dc2626",
  madd_2: "#dc2626",
  madd_4: "#b91c1c",
  madd_6: "#991b1b",

  ghunnah: "#16a34a",

  // make these clearly different
  ikhfa: "#d97706",      // orange
  idgham: "#7c3aed",    // purple

  qalqalah: "#2563eb",
  iqlab: "#0891b2",
  hamzat_wasl: "#6b7280",
  silent: "#111827",
  lam_shamsiyyah: "#0284c7",
  lam_qamariyyah: "#38bdf8",
};

function getRuleColor(rule?: string | null, apiColor?: string | null) {
  const key = String(rule || "").trim().toLowerCase();
  return ruleColors[key] || apiColor || fallbackColor;
}

const ruleLabels: Record<string, string> = {
  madd: "Madd",
  ghunnah: "Ghunnah",
  ikhfa: "Ikhfa",
  idgham: "Idgham",
  qalqalah: "Qalqalah",
  iqlab: "Iqlab",
};

const fallbackLegend = [
  { rule: "madd", label: "Madd", color: ruleColors.madd },
  { rule: "ghunnah", label: "Ghunnah", color: ruleColors.ghunnah },
  { rule: "ikhfa", label: "Ikhfa", color: ruleColors.ikhfa },
  { rule: "idgham", label: "Idgham", color: ruleColors.idgham },
  { rule: "qalqalah", label: "Qalqalah", color: ruleColors.qalqalah },
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
                  color: getRuleColor(seg.rule, seg.color),
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
      color: getRuleColor(rule, seg.color),
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
