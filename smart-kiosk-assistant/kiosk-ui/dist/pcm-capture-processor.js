// AudioWorklet processor that forwards raw Float32 mono PCM frames to the main
// thread. The main thread accumulates, resamples (if needed) and encodes WAV.
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      // Copy the channel-0 samples (Float32Array) out to the main thread.
      this.port.postMessage(input[0].slice(0));
    }
    return true; // keep processor alive
  }
}

registerProcessor('pcm-capture-processor', PCMCaptureProcessor);
