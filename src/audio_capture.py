import sounddevice as sd

def record_segment(sample_rate, duration, channels):
    
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()
    return audio_data.flatten()
