type ReferenceAudio = {
  available: boolean;
  url: string;
  reciter: string;
  surah: number;
  ayah: number;
};

export default function ReferenceAudioPlayer({
  audio,
}: {
  audio?: ReferenceAudio | null;
}) {
  if (!audio?.available || !audio.url) return null;

  return (
    <div className="mt-4 rounded-2xl border border-emerald-800/40 bg-emerald-950/30 p-4">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-lg">🔊</span>
        <p className="text-sm font-semibold text-emerald-300">
          Reference recitation — Surah {audio.surah}, Ayah {audio.ayah}
        </p>
      </div>
      <p className="mb-3 text-xs text-slate-400">
        Reciter: {audio.reciter.replaceAll("_", " ")}
      </p>
      <audio
        controls
        src={audio.url}
        className="w-full"
        preload="none"
      />
    </div>
  );
}