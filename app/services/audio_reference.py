from __future__ import annotations

DEFAULT_RECITER = "Abdul_Basit_Murattal_192kbps"

def get_reference_audio_url(
    surah: int,
    ayah: int,
    reciter: str = DEFAULT_RECITER,
) -> dict:
    surah_str = str(surah).zfill(3)
    ayah_str = str(ayah).zfill(3)
    url = f"https://everyayah.com/data/{reciter}/{surah_str}{ayah_str}.mp3"
    return {
        "available": True,
        "url": url,
        "reciter": reciter,
        "surah": surah,
        "ayah": ayah,
        "format": "mp3",
    }