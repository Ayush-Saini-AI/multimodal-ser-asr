import pandas as pd
import librosa
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

print("Loading dataset...")
df = pd.read_csv("meld_audio_subset/labels.csv")
pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

features = []

for index, row in df.iterrows():
    audio_path = f"meld_audio_subset/audio/{row['filename']}"
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Extract Acoustics
        rms = float(librosa.feature.rms(y=audio)[0].mean())
        tempo_result = librosa.beat.beat_track(y=audio, sr=sr)
        tempo = float(tempo_result[0].mean()) if hasattr(tempo_result[0], "__len__") else float(tempo_result[0])

        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
        pitch_vals = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if
                      magnitudes[:, t].max() > 0]
        pitch_mean = sum(pitch_vals) / len(pitch_vals) if pitch_vals else 0.0

        # Extract Sentiment
        sent_res = pipe(row['Utterance'][:512])[0]  # MELD uses 'Utterance' for the text
        sentiment_score = sent_res['score'] if sent_res['label'] == "POSITIVE" else (1.0 - sent_res['score'])

        features.append({
            "sentiment_score": sentiment_score,
            "pitch_mean": pitch_mean,
            "energy_mean": rms,
            "tempo": tempo,
            "emotion": row['Emotion'].upper()  # Format it for your UI
        })
        print(f"Processed {index + 1}/{len(df)}: {row['filename']}")
    except Exception as e:
        print(f"Skipping {row['filename']} due to error: {e}")

# Save the final training data
pd.DataFrame(features).to_csv("real_features.csv", index=False)
print("Done! real_features.csv created.")