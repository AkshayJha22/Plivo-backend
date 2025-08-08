from pydub import AudioSegment, silence
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import KMeans
import numpy as np, os, tempfile

def run_diarization(audio_path):
    audio = AudioSegment.from_file(audio_path)
    # split on silence (min_silence_len and silence_thresh may need tuning)
    chunks = silence.split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
    encoder = VoiceEncoder()
    embeddings = []
    temp_files = []

    for i, chunk in enumerate(chunks):
        tmpf = f"/tmp/segment_{i}.wav"
        chunk.export(tmpf, format="wav")
        temp_files.append(tmpf)
        wav = preprocess_wav(tmpf)
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)

    if len(embeddings) == 0:
        return []
    n_clusters = 2 if len(embeddings) >= 2 else 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.vstack(embeddings))
    labels = kmeans.labels_

    diarization = []
    for i, label in enumerate(labels):
        diarization.append({
            "segment_index": i,
            "speaker": f"speaker_{label+1}",
            "file": os.path.basename(temp_files[i])
        })

    # cleanup temp segment files (or keep for debugging)
    for f in temp_files:
        os.remove(f)
    return diarization
