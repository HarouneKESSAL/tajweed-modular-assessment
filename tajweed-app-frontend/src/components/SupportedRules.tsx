type SupportedRule = {
  name: string;
  module: string;
  status?: string;
};

export default function SupportedRules({
  items,
}: {
  items?: SupportedRule[] | null;
}) {
  if (!items || items.length === 0) return null;

  return (
    <div className="rounded-2xl bg-slate-800 p-4">
      <h3 className="mb-3 text-lg font-semibold">Supported rule types</h3>

      <p className="mb-3 text-sm text-slate-300">
        These are the Tajweed rules currently connected to trained acoustic
        modules. Other colored Mushaf rules can still appear as reference
        metadata.
      </p>

      <div className="flex flex-wrap gap-2">
        {items.map((item, index) => (
          <span
            key={`${item.name}-${index}`}
            className="rounded-full border border-slate-600 bg-slate-700 px-3 py-1 text-xs text-slate-100"
          >
            {item.name} · {item.module}
          </span>
        ))}
      </div>
    </div>
  );
}
