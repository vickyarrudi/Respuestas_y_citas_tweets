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


# ------------------- Helpers -------------------
def logout():
    st.session_state["logged_in"] = False
    st.rerun()


def login_page():
    st.title("Iniciar sesión")
    if st.button("Login demo"):
        st.session_state["logged_in"] = True
        st.rerun()


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


# =============================================================================
# App principal
# =============================================================================
def main_app():
    st.image("https://publicalab.com/assets/imgs/logo-publica-blanco.svg", width=200)
    st.markdown("<h1 class='big-title'> Análisis de respuestas y citas de Tweets </h1>", unsafe_allow_html=True)

    # --- Credenciales ---
    apify_token = read_secret_safe("apify_token", "APIFY_TOKEN")
    gemini_api_key = read_secret_safe("gemini_api_key", "GEMINI_API_KEY")

    if not apify_token:
        st.error("Falta APIFY_TOKEN.")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuración")
        url_input = st.text_input("URL del Tweet",
                                  placeholder="https://x.com/usuario/status/1234567890123456789")
        contexto = st.text_area("Contexto para análisis (opcional)",
                                help="Ej: Opiniones sobre un producto financiero")
        ejecutar = st.button("🚀 Ejecutar")

    # --- Inicializar clientes ---
    apify_client = ApifyClient(apify_token)

    model = None
    if gemini_api_key and genai is not None:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
            st.warning(f"No se pudo inicializar Gemini: {e}")
    else:
        st.info("Gemini no configurado. El análisis de IA será limitado.")

    # =============================================================================
    # Scrapers (usan apify_client desde outer scope, no como parámetro)
    # =============================================================================
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_replies(tweet_id: str) -> pd.DataFrame:
        try:
            if not tweet_id.isdigit():
                return pd.DataFrame()

            actor_id = "kaitoeasyapi/twitter-reply"
            run_input = {"conversation_ids": [tweet_id], "maxItems": 1000}
            run = apify_client.actor(actor_id).call(run_input=run_input)
            items = apify_client.dataset(run["defaultDatasetId"]).list_items().items or []
            df = pd.DataFrame(items)

            if "id" in df.columns:
                df = df[df["id"].astype(str) != str(tweet_id)]

            if "author" in df.columns:
                df["author/profilePicture"] = df["author"].apply(lambda x: x.get("profilePicture") if isinstance(x, dict) else None)
                df["author/followers"] = df["author"].apply(lambda x: x.get("followers") if isinstance(x, dict) else None)
                df["author/userName"] = df["author"].apply(lambda x: x.get("userName") if isinstance(x, dict) else None)

            for c in ["viewCount", "likeCount", "replyCount", "retweetCount", "quoteCount", "bookmarkCount", "author/followers"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if "createdAt" in df.columns:
                df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

            cols = ["author/profilePicture", "text", "createdAt", "author/userName", "author/followers",
                    "url", "likeCount", "replyCount", "retweetCount", "quoteCount", "bookmarkCount", "viewCount"]
            return _ensure_cols(df, cols).assign(tipo="reply")
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_quotes(tweet_id: str) -> pd.DataFrame:
        try:
            if not tweet_id.isdigit():
                return pd.DataFrame()

            actor_id = "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"
            run_input = {"filter:quote": True, "quoted_tweet_id": tweet_id, "maxItems": 1000}
            run = apify_client.actor(actor_id).call(run_input=run_input)
            items = apify_client.dataset(run["defaultDatasetId"]).list_items().items or []
            df = pd.DataFrame(items)

            if "author" in df.columns:
                df["author/profilePicture"] = df["author"].apply(lambda x: x.get("profilePicture") if isinstance(x, dict) else None)
                df["author/followers"] = df["author"].apply(lambda x: x.get("followers") if isinstance(x, dict) else None)
                df["author/userName"] = df["author"].apply(lambda x: x.get("userName") if isinstance(x, dict) else None)

            for c in ["viewCount", "likeCount", "replyCount", "retweetCount", "quoteCount", "bookmarkCount", "author/followers"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if "createdAt" in df.columns:
                df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

            cols = ["author/profilePicture", "text", "createdAt", "author/userName", "author/followers",
                    "url", "likeCount", "replyCount", "retweetCount", "quoteCount", "bookmarkCount", "viewCount"]
            return _ensure_cols(df, cols).assign(tipo="quote")
        except Exception:
            return pd.DataFrame()

    # =============================================================================
    # IA
    # =============================================================================
    def clasificar_tweet(texto: str, contexto: str) -> str:
        if not model or not texto.strip():
            return "NEUTRO"
        prompt = f"CONTEXTO: {contexto}\nClasifica este tweet en POSITIVO, NEGATIVO o NEUTRO.\nTweet: \"{texto}\""
        try:
            resp = model.generate_content(prompt, generation_config={"temperature": 0.2})
            return (resp.text or "").strip().upper()[:8] or "NEUTRO"
        except Exception:
            return "NEUTRO"

    def extraer_temas_con_ia(textos: list[str], sentimiento: str, contexto: str, num_temas: int = 3) -> str:
        if not model:
            return "IA no disponible."
        textos = [t for t in textos if t.strip()]
        if not textos:
            return "No hay tweets suficientes."
        texto_join = "\n".join(textos[:500])
        prompt = f"CONTEXTO: {contexto}\nExtrae {num_temas} temas principales de tweets {sentimiento}.\nTweets:\n{texto_join}"
        try:
            resp = model.generate_content(prompt, generation_config={"temperature": 0.4})
            return (resp.text or "").strip()
        except Exception as e:
            return f"Error: {e}"

    # =============================================================================
    # Flujo principal
    # =============================================================================
    tweet_id = extract_tweet_id_from_url(url_input) if url_input else None

    if tweet_id and ejecutar:
        st.subheader("📥 Descargando datos de X…")
        df_replies, df_quotes = get_replies(tweet_id), get_quotes(tweet_id)
        st.success(f"✅ {len(df_replies)} respuestas y {len(df_quotes)} citas descargadas.")

        # Métricas
        df_all = pd.concat([df_replies, df_quotes], ignore_index=True)
        total_views = pd.to_numeric(df_all.get("viewCount", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()
        interaction_cols = ["likeCount", "replyCount", "retweetCount", "quoteCount", "bookmarkCount"]
        total_interacciones = df_all[interaction_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum() if any(c in df_all for c in interaction_cols) else 0

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

        # ---------- Botón de análisis ----------
        st.subheader("🤖 Análisis con Gemini")
        if st.button("Analizar Conversación con IA"):
            if model:
                df_todos = df_all.copy()
                textos_todos = df_todos["text"].dropna().astype(str).tolist()
                with st.spinner("Extrayendo temas principales…"):
                    resultados = extraer_temas_con_ia(textos_todos, "mixto", contexto, 5)
                st.markdown("### Temas principales detectados")
                st.write(resultados)

                # Clasificación de sentimientos
                from concurrent.futures import ThreadPoolExecutor, as_completed
                resultados_sent = ["NEUTRO"] * len(df_todos)
                tweets_validos = [(i, t) for i, t in enumerate(df_todos["text"]) if pd.notna(t) and str(t).strip()]
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(clasificar_tweet, t, contexto): i for i, t in tweets_validos}
                    for f in as_completed(futures):
                        resultados_sent[futures[f]] = f.result() if f.result() else "NEUTRO"
                df_todos["sentimiento"] = resultados_sent

                # Pie chart sentimientos
                counts = df_todos["sentimiento"].value_counts().reset_index()
                counts.columns = ["Sentimiento", "Cantidad"]
                fig = px.pie(counts, values="Cantidad", names="Sentimiento", title="Distribución de Sentimientos")
                st.plotly_chart(fig, use_container_width=True)

                # Evolución temporal
                if "createdAt" in df_todos.columns:
                    df_time = df_todos.dropna(subset=["createdAt"]).copy()
                    df_time["createdAt"] = pd.to_datetime(df_time["createdAt"], errors="coerce", utc=True)
                    if not df_time.empty:
                        timeline = df_time.groupby(df_time["createdAt"].dt.date).size().reset_index(name="Cantidad")
                        fig_tl = px.line(timeline, x="createdAt", y="Cantidad", markers=True,
                                         title="📈 Evolución de Tweets en el Tiempo")
                        st.plotly_chart(fig_tl, use_container_width=True)
            else:
                st.error("Configura GEMINI_API_KEY para análisis con IA.")

    elif url_input:
        st.error("No pude extraer un ID válido de esa URL. Revisa que tenga el formato /status/<número>.")


# --- Entrada ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = True

if st.session_state["logged_in"]:
    main_app()
else:
    login_page()
