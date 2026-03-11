import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import soundfile as sf
import io
import json
from pathlib import Path
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marine Acoustic Explorer",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constantes ─────────────────────────────────────────────────────────────────
LABELS = [
    "🐋 Cetáceo (ballena/delfín)",
    "🐬 Delfín (clicks)",
    "🐟 Sonido biótico (pez/otro)",
    "🚢 Barco/motor",
    "🌊 Océano/fondo ambiental",
    "⚡ Transiente desconocido",
    "🔇 Silencio",
]

CONFIDENCE = ["Seguro", "Probable", "Dudoso"]

FREQ_REFS = {
    "🐋 Ballena jorobada": (20, 4000),
    "🐬 Delfín":           (2000, 150000),
    "🚢 Barco/motor":      (0, 1000),
    "🐟 Peces":            (100, 3000),
}

FREQ_COLORS = {
    "🐋 Ballena jorobada": "#4fc3f7",
    "🐬 Delfín":           "#81c784",
    "🚢 Barco/motor":      "#ef5350",
    "🐟 Peces":            "#ffb74d",
}

LABELS_CSV = Path("labels.csv")

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_clips():
    candidates = [
        Path("clips_with_clusters_perch.parquet"),
        Path("/content/hackathon-participants/outputs/clips_with_clusters_perch.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    st.error("No se encontró clips_with_clusters_perch.parquet. Asegúrate de haber corrido el pipeline.")
    st.stop()

def _get_audio_from_r2(file_path, start_s, end_s):
    """Descarga audio desde R2 cuando no está disponible localmente."""
    import boto3, tempfile, os
    # Reconstruir el key en R2 a partir del path local
    # Rutas locales: /content/hackathon-participants/hackathon_data/marine-acoustic/marine-acoustic/...
    # En R2 el dataset marine-acoustic-core tiene la misma estructura
    parts = Path(file_path).parts
    try:
        idx = list(parts).index("marine-acoustic")
        key = "marine-acoustic-core/" + "/".join(parts[idx+1:])
    except ValueError:
        key = Path(file_path).name

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("R2_ENDPOINT"),
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
    )
    bucket = os.environ.get("R2_BUCKET", "sala-2026-hackathon-data")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        s3.download_fileobj(bucket, key, tmp)
        tmp_path = tmp.name
    audio, sr = librosa.load(tmp_path, sr=None, offset=float(start_s),
                             duration=float(end_s - start_s), mono=True)
    os.unlink(tmp_path)
    return audio, sr

@st.cache_data
def load_audio_clip(file_path, start_s, end_s):
    if Path(file_path).exists():
        audio, sr = librosa.load(
            file_path, sr=None, offset=float(start_s),
            duration=float(end_s - start_s), mono=True
        )
    else:
        audio, sr = _get_audio_from_r2(file_path, start_s, end_s)
    return audio, sr

def make_spectrogram(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma")

    # Eje Y en kHz
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f} kHz")
    )
    ax.set_xlabel("Tiempo (s)", color="white")
    ax.set_ylabel("Frecuencia", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")

    # Bandas de referencia
    nyquist = sr / 2
    for name, (fmin, fmax) in FREQ_REFS.items():
        fmax_clipped = min(fmax, nyquist)
        if fmax_clipped > fmin:
            ax.axhspan(
                fmin, fmax_clipped,
                alpha=0.10,
                color=FREQ_COLORS[name],
                label=f"{name} ({fmin/1000:.1f}–{fmax_clipped/1000:.1f} kHz)",
            )

    ax.legend(
        loc="upper right", fontsize=7,
        facecolor="#1e1e2e", labelcolor="white",
        framealpha=0.8,
    )
    plt.tight_layout()
    return fig

def audio_to_bytes(audio, sr):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf

def load_labels():
    if LABELS_CSV.exists():
        df = pd.read_csv(LABELS_CSV)
        return df.set_index("clip_id").to_dict(orient="index")
    return {}

def save_label(clip_id, labels, confidence, notes, flag_expert, annotator):
    existing = load_labels()
    existing[clip_id] = {
        "clip_id":     clip_id,
        "labels":      json.dumps(labels, ensure_ascii=False),
        "confidence":  confidence,
        "notes":       notes,
        "flag_expert": flag_expert,
        "annotator":   annotator,
        "timestamp":   datetime.now().isoformat(),
    }
    pd.DataFrame(existing.values()).to_csv(LABELS_CSV, index=False)

# ── Session state ──────────────────────────────────────────────────────────────
if "clip_idx" not in st.session_state:
    st.session_state.clip_idx = 0
if "saved_labels" not in st.session_state:
    st.session_state.saved_labels = load_labels()

# ── Cargar datos ───────────────────────────────────────────────────────────────
clips_df = load_clips()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🐋 Marine Acoustic\nExplorer")
    st.caption("Herramienta de etiquetado para investigadores")
    st.divider()

    annotator = st.text_input("Tu nombre", placeholder="ej. María García")

    st.divider()
    st.markdown("**Filtros**")

    cluster_opts = ["Todos"] + [str(c) for c in sorted(clips_df["cluster"].unique())]
    cluster_filter = st.selectbox("Cluster", cluster_opts)

    label_filter = st.selectbox(
        "Mostrar",
        ["Sin etiquetar primero", "Todos", "Ya etiquetados", "Marcados para experto"],
    )

    st.divider()
    n_labeled = len(st.session_state.saved_labels)
    n_total   = len(clips_df)
    pct = n_labeled / n_total if n_total else 0
    st.metric("Progreso", f"{n_labeled} / {n_total} clips")
    st.progress(pct)

    # Distribución de etiquetas
    if st.session_state.saved_labels:
        all_labels = []
        for v in st.session_state.saved_labels.values():
            try:
                all_labels.extend(json.loads(v.get("labels", "[]")))
            except Exception:
                pass
        if all_labels:
            st.caption("Distribución de etiquetas:")
            counts = pd.Series(all_labels).value_counts()
            st.bar_chart(counts)

        flagged = sum(
            1 for v in st.session_state.saved_labels.values()
            if v.get("flag_expert")
        )
        if flagged:
            st.warning(f"🔍 {flagged} clips marcados para experto")

    st.divider()
    if st.session_state.saved_labels:
        df_exp = pd.DataFrame(st.session_state.saved_labels.values())
        st.download_button(
            "⬇ Descargar etiquetas CSV",
            df_exp.to_csv(index=False),
            file_name="labels_marine.csv",
            mime="text/csv",
        )

    # Guía rápida
    st.divider()
    with st.expander("📖 Guía rápida"):
        st.markdown("""
**Cómo usar:**
1. Escribe tu nombre
2. Presiona ▶ en el audio para escuchar
3. Observa el espectrograma
4. Selecciona una o más etiquetas
5. Indica tu confianza
6. Agrega notas si ves algo interesante
7. Guarda y continúa

**En el espectrograma:**
- Eje X = tiempo
- Eje Y = frecuencia (kHz)
- Colores más brillantes = más intenso

**Rangos de referencia:**
- 🔵 Ballena jorobada: 20 Hz–4 kHz
- 🟢 Delfín: 2–150 kHz
- 🔴 Barco: < 1 kHz
- 🟠 Peces: 100 Hz–3 kHz
        """)

# ── Filtrar clips ──────────────────────────────────────────────────────────────
filtered = clips_df.copy()

if cluster_filter != "Todos":
    filtered = filtered[filtered["cluster"] == int(cluster_filter)]

labeled_ids = set(int(k) for k in st.session_state.saved_labels.keys())

if label_filter == "Sin etiquetar primero":
    unlabeled = filtered[~filtered["clip_id"].isin(labeled_ids)]
    labeled   = filtered[ filtered["clip_id"].isin(labeled_ids)]
    filtered  = pd.concat([unlabeled, labeled]).reset_index(drop=True)
elif label_filter == "Ya etiquetados":
    filtered = filtered[filtered["clip_id"].isin(labeled_ids)].reset_index(drop=True)
elif label_filter == "Marcados para experto":
    expert_ids = {
        int(k) for k, v in st.session_state.saved_labels.items()
        if v.get("flag_expert")
    }
    filtered = filtered[filtered["clip_id"].isin(expert_ids)].reset_index(drop=True)

if filtered.empty:
    st.info("No hay clips con ese filtro.")
    st.stop()

st.session_state.clip_idx = min(st.session_state.clip_idx, len(filtered) - 1)
row = filtered.iloc[st.session_state.clip_idx]

# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_nav = st.columns([3, 1])

with col_title:
    already = int(row.clip_id) in labeled_ids
    badge = "✅ Etiquetado" if already else "⬜ Sin etiquetar"
    st.markdown(f"### Clip #{int(row.clip_id)}  `{Path(row.file_path).name}`  {badge}")
    st.caption(
        f"Cluster: **{row.cluster}** | "
        f"Tiempo: {row.start_s:.0f}s – {row.end_s:.0f}s | "
        f"Unidad: `{row.unit}` | "
        f"Grupo: `{row.source_group}`"
    )

with col_nav:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Ant", disabled=st.session_state.clip_idx == 0):
            st.session_state.clip_idx -= 1
            st.rerun()
    with c2:
        if st.button("Sig →", disabled=st.session_state.clip_idx >= len(filtered) - 1):
            st.session_state.clip_idx += 1
            st.rerun()
    st.caption(f"{st.session_state.clip_idx + 1} / {len(filtered)}")

# ── Audio + Espectrograma ──────────────────────────────────────────────────────
try:
    audio, sr = load_audio_clip(row.file_path, row.start_s, row.end_s)

    col_spec, col_ref = st.columns([3, 1])

    with col_spec:
        st.markdown("**Espectrograma**")
        fig = make_spectrogram(audio, sr)
        st.pyplot(fig)
        plt.close(fig)
        st.audio(audio_to_bytes(audio, sr), format="audio/wav")

    with col_ref:
        st.markdown("**Referencia de frecuencias**")
        st.markdown("""
| Animal | Rango |
|--------|-------|
| 🐋 Ballena jorobada | 20 Hz – 4 kHz |
| 🐬 Delfín | 2 – 150 kHz |
| 🚢 Barco | < 1 kHz |
| 🐟 Peces | 100 Hz – 3 kHz |
        """)
        st.divider()
        st.markdown("**Info del clip**")
        st.markdown(f"""
- **SR:** {sr/1000:.0f} kHz
- **Duración:** {row.end_s - row.start_s:.0f} s
- **Max freq:** {sr/2000:.0f} kHz
        """)

except Exception as e:
    st.error(f"Error cargando audio: {e}")
    st.info("Verifica que el archivo existe en la ruta indicada.")

st.divider()

# ── Etiquetado ─────────────────────────────────────────────────────────────────
st.markdown("### ¿Qué escuchas en este clip?")
st.caption("Puedes seleccionar varias etiquetas a la vez")

existing = st.session_state.saved_labels.get(int(row.clip_id), {})
existing_labels = json.loads(existing.get("labels", "[]")) if existing else []

selected_labels = st.multiselect(
    "Etiquetas",
    LABELS,
    default=[l for l in existing_labels if l in LABELS],
    label_visibility="collapsed",
)

col_conf, col_flag = st.columns(2)

with col_conf:
    prev_conf = existing.get("confidence", "Probable") if existing else "Probable"
    conf_idx  = CONFIDENCE.index(prev_conf) if prev_conf in CONFIDENCE else 1
    confidence = st.radio(
        "Nivel de confianza",
        CONFIDENCE,
        index=conf_idx,
        horizontal=True,
    )

with col_flag:
    st.markdown(" ")
    flag_expert = st.checkbox(
        "🔍 Marcar para revisión de experto",
        value=bool(existing.get("flag_expert", False)) if existing else False,
    )

notes = st.text_area(
    "Notas (opcional)",
    value=existing.get("notes", "") if existing else "",
    placeholder="ej. 'Click repetitivo en segundo 3, posible delfín ~8 kHz. Señal débil.'",
    height=90,
)

col_btn, col_msg = st.columns([1, 3])

with col_btn:
    save_disabled = not annotator or not selected_labels
    if st.button("💾 Guardar y continuar", type="primary", disabled=save_disabled):
        save_label(
            int(row.clip_id), selected_labels, confidence,
            notes, flag_expert, annotator
        )
        st.session_state.saved_labels = load_labels()
        if st.session_state.clip_idx < len(filtered) - 1:
            st.session_state.clip_idx += 1
        st.rerun()

with col_msg:
    if not annotator:
        st.warning("Ingresa tu nombre en el panel izquierdo para guardar.")
    elif not selected_labels:
        st.info("Selecciona al menos una etiqueta.")
    elif int(row.clip_id) in labeled_ids:
        prev = st.session_state.saved_labels[int(row.clip_id)]
        st.success(
            f"✅ Etiquetado por **{prev.get('annotator','?')}** — "
            f"{prev.get('confidence','?')} | "
            f"{', '.join(json.loads(prev.get('labels','[]')))}"
        )
