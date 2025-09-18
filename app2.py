import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from apify_client import ApifyClient

# Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ================================================================
# Helpers
# ================================================================
def read_secret_safe(key: str, env_key: str):
    val = st.session_state.get(key)
    if val:
        return val
    val = os.getenv(env_key)
    if val:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return None

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]]

_TWEET_ID_PATTERNS = [
    re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/[^/]+/status/(\d+)", re.I),
    re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/i/(?:web/)?status/(\d+)", re.I),
]

def extract_tweet_id_from_url(value: str | None) -> str | None:
    if not value:
        return None
    s = value.strip()
    if s.isdigit():
        return s
    for rx in _TWEET_ID_PATTERNS:
        m = rx.search(s)
        if m:
            return m.group(1)
    return None

# ================================================================
# Scrapers
# ================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_replies(apify_client, tweet_id: str) -> pd.DataFrame:
    try:
        actor_id = "kaitoeasyapi/twitter-reply"
        run_input = {"conversation_ids": [str(tweet_id)], "maxItems": 1000}
        run = apify_client.actor(actor_id).call(run_input=run_input)
        items = apify_client.dataset(run["defaultDatasetId"]).list_items().items or []
        df = pd.DataFrame(items)

        if df.empty:
            return df

        if "author" in df.columns:
            df["author/profilePicture"] = df["author"].apply(lambda x: x.get("profilePicture") if isinstance(x, dict) else None)
            df["author/followers"] = df["author"].apply(lambda x: x.get("followers") if isinstance(x, dict) else None)
            df["author/userName"] = df["author"].apply(lambda x: x.get("userName") if isinstance(x, dict) else None)

        for c in ["viewCount","likeCount","replyCount","retweetCount","quoteCount","bookmarkCount","author/followers"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        if "createdAt" in df.columns:
            df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

        cols = [
            "author/profilePicture","text","createdAt","author/userName","author/followers","url","likeCount",
            "replyCount","retweetCount","quoteCount","bookmarkCount","viewCount"
        ]
        df = _ensure_cols(df, cols)
        df["tipo"] = "reply"
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_quotes(apify_client, tweet_id: str) -> pd.DataFrame:
    try:
        actor_id = "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"
        run_input = {"filter:quote": True, "quoted_tweet_id": str(tweet_id), "maxItems": 1000}
        run = apify_client.actor(actor_id).call(run_input=run_input)
        items = apify_client.dataset(run["defaultDatasetId"]).list_items().items or []
        df = pd.DataFrame(items)

        if df.empty:
            return df

        if "author" in df.columns:
            df["author/profilePicture"] = df["author"].apply(lambda x: x.get("profilePicture") if isinstance(x, dict) else None)
            df["author/followers"] = df["author"].apply(lambda x: x.get("followers") if isinstance(x, dict) else None)
            df["author/userName"] = df["author"].apply(lambda x: x.get("userName") if isinstance(x, dict) else None)

        for c in ["viewCount","likeCount","replyCount","retweetCount","quoteCount","bookmarkCount","author/followers"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        if "createdAt" in df.columns:
            df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

        cols = [
            "author/profilePicture","text","createdAt","author/userName","author/followers","url","likeCount",
            "replyCount","retweetCount","quoteCount","bookmarkCount","viewCount"
        ]
        df = _ensure_cols(df, cols)
        df["tipo"] = "quote"
        return df
    except Exception:
        return pd.DataFrame()

# ================================================================
# IA
# ================================================================
def clasificar_tweet(model, texto: str, contexto: str) -> str:
    if not model or not isinstance(texto, str) or not texto.strip():
        return "NEUTRO"
    prompt = (
        f"CONTEXTO: {contexto}\n"
        "Clasifica el sentimiento del siguiente tweet en POSITIVO, NEGATIVO o NEUTRO.\n"
        "Responde únicamente con POSITIVO, NEGATIVO o NEUTRO.\n"
        f'Tweet: "{texto}"\nSentimiento:'
    )
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0.2})
        return (resp.text or "").strip().upper()[:8] or "NEUTRO"
    except Exception:
        return "NEUTRO"

def extraer_temas_con_ia(model, textos: list[str], sentimiento: str, contexto: str, num_temas: int = 3) -> str:
    if not model:
        return "El modelo de IA no está disponible para extraer temas."
    textos = [t for t in textos if isinstance(t, str) and t.strip()]
    if not textos:
        return "No hay tweets suficientes para extraer temas."
    texto_join = "\n".join(textos[:500])
    prompt = f"""CONTEXTO: {contexto}
Aquí hay tweets clasificados como {sentimiento}. Extrae los {num_temas} temas principales, explicando brevemente cada uno y dando un ejemplo.
Formato:
Tema: [nombre del tema 1]
Explicación: [breve explicación]
Ejemplo: "[tweet de ejemplo relevante]"

Tema: [nombre del tema 2]
Explicación: [breve explicación]
Ejemplo: "[tweet de ejemplo relevante]"

Tweets:
{texto_join}
"""
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0.4})
        return (resp.text or "").strip()
    except Exception as e:
        return f"No se pudieron extraer temas. Error: {e}"

# ================================================================
# Main App
# ================================================================
def main_app():
    st.image("https://publicalab.com/assets/imgs/logo-publica-blanco.svg", width=200)
    st.markdown("<h1 class='big-title'> Análisis de respuestas y citas de Tweets </h1>", unsafe_allow_html=True)

    # Credenciales
    apify_token    = read_secret_safe("apify_token", "APIFY_TOKEN")
    gemini_api_key = read_secret_safe("gemini_api_key", "GEMINI_API_KEY")
    if not apify_token:
        st.error("Falta APIFY_TOKEN.")
        st.stop()
    apify_client = ApifyClient(apify_token)

    model = None
    if gemini_api_key and genai is not None:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
            st.warning(f"No se pudo inicializar Gemini: {e}")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        url_input = st.text_input("URL del Tweet")
        contexto = st.text_area("Contexto para el análisis (opcional)")
        ejecutar = st.button("🚀 Ejecutar")

    # Estado
    for key, default in [
        ("tweet_id", None),
        ("df_replies", pd.DataFrame()),
        ("df_quotes", pd.DataFrame()),
        ("data_loaded", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    parsed_id = extract_tweet_id_from_url(url_input) if url_input else None

    # Ejecutar scraping
    if ejecutar:
        if not parsed_id:
            st.error("No se pudo extraer un ID válido de esa URL.")
            st.stop()

        st.session_state["tweet_id"] = parsed_id
        st.subheader("📥 Descargando datos de X…")

        df_replies = get_replies(apify_client, parsed_id)
        df_quotes = get_quotes(apify_client, parsed_id)

        st.session_state["df_replies"] = df_replies
        st.session_state["df_quotes"] = df_quotes
        st.session_state["data_loaded"] = True

        st.success(f"✅ {len(df_replies)} respuestas y {len(df_quotes)} citas descargadas.")

    # Mostrar datos
    if st.session_state["data_loaded"]:
        df_replies = st.session_state["df_replies"]
        df_quotes = st.session_state["df_quotes"]
        df_all = pd.concat([df_replies, df_quotes], ignore_index=True)

        # --- Métricas ---
        views_col = next((c for c in ["viewCount","views","impressions"] if c in df_all.columns), None)
        total_views = df_all[views_col].sum() if views_col else 0
        interaction_cols = ["likeCount","replyCount","retweetCount","quoteCount","bookmarkCount"]
        present_cols = [c for c in interaction_cols if c in df_all.columns]
        total_interacciones = df_all[present_cols].sum().sum() if present_cols else 0

        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 1.5em; font-weight: bold; color: #FFFFFF;">
                📈 Alcance Total: {int(total_views):,} visualizaciones
            </div>
            <div style="font-size: 1.5em; font-weight: bold; color: #FFFFFF;">
                💬 Interacciones Totales: {int(total_interacciones):,}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Previews ---
        if not df_replies.empty:
            st.write("### Algunas Respuestas")
            st.data_editor(df_replies.head(5), column_config={
                "author/profilePicture": st.column_config.ImageColumn("Foto"),
                "url": st.column_config.LinkColumn("Tweet"),
                "author/userName": st.column_config.TextColumn("Usuario"),
                "text": st.column_config.TextColumn("Contenido"),
            }, hide_index=True, use_container_width=True)
        if not df_quotes.empty:
            st.write("### Algunas Citas")
            st.data_editor(df_quotes.head(5), column_config={
                "author/profilePicture": st.column_config.ImageColumn("Foto"),
                "url": st.column_config.LinkColumn("Tweet"),
                "author/userName": st.column_config.TextColumn("Usuario"),
                "text": st.column_config.TextColumn("Contenido"),
            }, hide_index=True, use_container_width=True)

        # --- Botón análisis ---
        st.subheader("🤖 Análisis con Gemini")
        if st.button("Analizar Conversación con IA"):
            df_todos = df_all.copy()
            if df_todos.empty:
                st.warning("No hay tweets para analizar.")
                st.stop()

            # Temas generales
            with st.spinner("Extrayendo temas principales…"):
                textos = df_todos["text"].dropna().astype(str).tolist()
                resultados = extraer_temas_con_ia(model, textos, "mixto", contexto, num_temas=5)
                st.write(resultados)

            # Sentimientos
            from concurrent.futures import ThreadPoolExecutor, as_completed
            resultados_sent = ["NEUTRO"] * len(df_todos)
            validos = [(i,t) for i,t in enumerate(df_todos["text"]) if pd.notna(t) and str(t).strip()]
            if validos:
                progress_bar = st.progress(0)
                total = len(validos)
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(clasificar_tweet, model, t, contexto): i for i,t in validos}
                    for done, f in enumerate(as_completed(futures), 1):
                        idx = futures[f]
                        try:
                            resultados_sent[idx] = f.result()
                        except Exception:
                            resultados_sent[idx] = "NEUTRO"
                        progress_bar.progress(done/total)
                df_todos["sentimiento"] = resultados_sent
                st.success("✅ Clasificación completada.")

            # Distribución de sentimientos
            if "sentimiento" in df_todos.columns:
                counts = df_todos["sentimiento"].value_counts().reset_index()
                counts.columns = ["Sentimiento","Cantidad"]
                counts["Porcentaje"] = counts["Cantidad"]/counts["Cantidad"].sum()*100
                st.dataframe(counts, use_container_width=True, hide_index=True)
                fig = px.pie(counts, values="Cantidad", names="Sentimiento")
                st.plotly_chart(fig, use_container_width=True)

            # Timeline
            if "createdAt" in df_todos.columns:
                df_time = df_todos.dropna(subset=["createdAt"]).copy()
                df_time["createdAt"] = pd.to_datetime(df_time["createdAt"])
                df_time["date"] = df_time["createdAt"].dt.date
                timeline = df_time.groupby("date").size().reset_index(name="Tweets")
                fig_tl = px.line(timeline, x="date", y="Tweets", markers=True)
                st.plotly_chart(fig_tl, use_container_width=True)

# ================================================================
# Entry
# ================================================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = True

if st.session_state["logged_in"]:
    main_app()
