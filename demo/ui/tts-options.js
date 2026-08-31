// @ts-check

export const TTS_BACKENDS = Object.freeze([
  { id: "silero", label: "Silero v5.5 RU" },
  { id: "rhvoice", label: "RHVoice" },
]);

const TTS_VOICES = Object.freeze({
  silero: Object.freeze([
    { id: "xenia", label: "Xenia" },
    { id: "kseniya", label: "Kseniya" },
    { id: "aidar", label: "Aidar" },
    { id: "eugene", label: "Eugene" },
    { id: "baya", label: "Baya" },
  ]),
  rhvoice: Object.freeze([
    { id: "Aleksandr", label: "Александр" },
    { id: "Mikhail", label: "Михаил" },
    { id: "Evgeniy-Rus", label: "Евгений" },
    { id: "Pavel", label: "Павел" },
  ]),
});

/** @param {string} backend */
export function voicesForTtsBackend(backend) {
  return TTS_VOICES[backend] || TTS_VOICES.silero;
}

/** @param {string} backend @param {string} voice */
export function encodeTtsSelection(backend, voice) {
  const knownBackend = TTS_BACKENDS.some((item) => item.id === backend) ? backend : "silero";
  const voices = voicesForTtsBackend(knownBackend);
  const knownVoice = voices.some((item) => item.id === voice) ? voice : voices[0].id;
  return `${knownBackend}:${knownVoice}`;
}

/** @param {string | null | undefined} raw */
export function decodeTtsSelection(raw) {
  if (raw && raw.includes(":")) {
    const [backend, voice] = raw.split(":", 2);
    const [validBackend, validVoice] = encodeTtsSelection(backend, voice).split(":", 2);
    return { backend: validBackend, voice: validVoice };
  }
  for (const backend of TTS_BACKENDS) {
    if (voicesForTtsBackend(backend.id).some((voice) => voice.id === raw)) {
      return { backend: backend.id, voice: raw };
    }
  }
  return { backend: "silero", voice: "xenia" };
}
