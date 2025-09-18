# streamlit_app.py
import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from apify_client import ApifyClient

# ---------- Config de página ----------
st.set_page_config(page_title="Twitter Scraper · Respuestas y Citas", page_icon="📊", layout="wide")

# ---------- Gemini (opcional) ----------
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ===================== Helpers =====================
def read_secret_safe(key: str, env_key: str):
    """Busca credenciales en session_state, env y st.secrets."""
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
    if df.empty:
        return df
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

# Evita UnhashableParamError: cacheamos el CLIENT como recurso, data por separado
@st.cache_resource(show_spinner=False)
def get_apify_client(token: str) -> ApifyClient:
    return ApifyClient(token)

# ===================== Scrapers (cacheados por tweet_id + token) =====================
@st.cache_data(ttl=3600, show_spinner=False)
def get_replies(tweet_id: str, token: str) -> pd.DataFrame:
    """Obtiene replies del hilo (conversation_id=tweet_id)."""
    try:
        if not (isinstance(tweet_id, str) and tweet_id.isdigit()):
            return pd.DataFrame()
        client = get_apify_client(token)
        run = client.actor("kaitoeasyapi/twitter-reply").call(run_input={
            "conversation_ids": [tweet_id],
            "maxItems": 3000
        })
        items = client.dataset(run["defaultDatasetId"]).list_items().items or []
        df = pd.DataFrame(items)
        if df.empty:
            return df

        # Derivados del autor
        if "author" in df.columns:
            df["author/profilePicture"] = df["author"].apply(lambda x: x.get("profilePicture") if isinstance(x, dict) else None)
            df["author/followers"] = df["author"].apply(lambda x: x.get("followers") if isinstance(x, dict) else None)
            df["author/userName"] = df["author"].apply(lambda x: x.get("userName") if isinstance(x, dict) else None)

        # Numéricos y fechas
        for c in ["viewCount","likeCount","replyCount","retweetCount","quoteCount","bookmarkCount","author/followers"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if "createdAt" in df.columns:
            df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

        cols = ["author/profilePicture","text","createdAt","author/userName","author/followers",
                "url","likeCount","replyCount","retweetCount","quoteCount","bookmarkCount","viewCount","id"]
        df = _ensure_cols(df, cols)
        df["tipo"] = "reply"
        return df
    except Exception as e:
        st.error(f"Error al obtener respuestas: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_quotes(tweet_id: str, token: str) -> pd.DataFrame:
    """Obtiene quote tweets que citan el tweet_id."""
    try:
        if not (isinstance(tweet_id, str) and tweet_id.isdigit()):
            return pd.DataFrame()
        client = get_apify_client(token)
        run = client.actor("kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input={
            "filter:quote": True,
            "quoted_tweet_id": str(tweet_id),
            "maxItems": 3000
        })
        items = client.dataset(run["defaultDatasetId"]).list_items().items or []
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

        cols = ["author/profilePicture","text","createdAt","author/userName","author/followers",
                "url","likeCount","replyCount","retweetCount","quoteCount","bookmarkCount","viewCount","id"]
        df = _ensure_cols(df, cols)
        df["tipo"] = "quote"
        return df
    except Exception as e:
        st.error(f"Error al obtener citas: {e}")
        return pd.DataFrame()

# ===================== IA =====================
def build_gemini_model(api_key: str | None):
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        st.warning(f"No se pudo inicializar Gemini: {e}")
        return None

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

def extraer_temas_con_ia(model, textos: list[str], sentimiento: str, contexto: str, num_temas: int = 5) -> str:
    if not model:
        return "El modelo de IA no está disponible."
    textos = [t for t in textos if isinstance(t, str) and t.strip()]
    if not textos:
        return "No hay tweets suficientes."
    texto_join = "\n".join(textos[:500])
    prompt = f"""CONTEXTO: {contexto}
Aquí hay tweets clasificados como {sentimiento}. Extrae {num_temas} temas principales. 
Tweets:
{texto_join}"""
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0.4})
        return (resp.text or "").strip()
    except Exception as e:
        return f"No se pudieron extraer temas. Error: {e}"

# ===================== App =====================
def main_app():
    st.image("https://publicalab.com/assets/imgs/logo-publica-blanco.svg", width=200)
    st.markdown("<h1 class='big-title'> Análisis de Respuestas y Citas - X </h1>", unsafe_allow_html=True)

    # Credenciales
    apify_token = read_secret_safe("apify_token", "APIFY_TOKEN")
    gemini_api_key = read_secret_safe("gemini_api_key", "GEMINI_API_KEY")
    if not apify_token:
        st.error("Falta APIFY_TOKEN (env o .streamlit/secrets.toml).")
        st.stop()

    model = build_gemini_model(gemini_api_key)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        url_input = st.text_input(
            "URL del Tweet",
            placeholder="https://x.com/usuario/status/1234567890123456789",
            help="Pega la URL completa del tweet."
        )
        contexto = st.text_area("Contexto (opcional)", help="Ej: Opiniones sobre producto X.")
        ejecutar = st.button("🚀 Ejecutar")

    # Estado persistente
    for k, v in [
        ("tweet_id", None),
        ("df_replies", pd.DataFrame()),
        ("df_quotes", pd.DataFrame()),
        ("data_loaded", False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    parsed_id = extract_tweet_id_from_url(url_input) if url_input else None

    # ---------- Ejecutar descarga ----------
    if ejecutar:
        if not parsed_id:
            st.error("No pude extraer un ID válido de esa URL. Debe tener /status/<número>.")
            st.stop()
        st.session_state["tweet_id"] = parsed_id

        st.subheader("📥 Descargando datos de X/Twitter…")
        df_replies = get_replies(parsed_id, apify_token)
        df_quotes  = get_quotes(parsed_id, apify_token)

        # Limpieza básica (excluir original si aparece y duplicados por URL)
        if "id" in df_replies.columns:
            df_replies = df_replies[df_replies["id"].astype(str) != str(parsed_id)]
        if "id" in df_quotes.columns:
            df_quotes  = df_quotes[df_quotes["id"].astype(str) != str(parsed_id)]
        if "url" in df_replies.columns:
            df_replies = df_replies.drop_duplicates(subset=["url"]).reset_index(drop=True)
        if "url" in df_quotes.columns:
            df_quotes  = df_quotes.drop_duplicates(subset=["url"]).reset_index(drop=True)

        st.session_state["df_replies"] = df_replies
        st.session_state["df_quotes"]  = df_quotes
        st.session_state["data_loaded"] = True

        st.success(f"✅ {len(df_replies)} respuestas y {len(df_quotes)} citas descargadas.")

    # ---------- Mostrar datos si están cargados ----------
    if st.session_state["data_loaded"]:
        df_replies = st.session_state["df_replies"]
        df_quotes  = st.session_state["df_quotes"]
        df_all = pd.concat([df_replies, df_quotes], ignore_index=True)

        # Métricas de alcance e interacciones
        candidate_view_cols = ["viewCount", "views", "impressions", "impression_count", "public_metrics.impression_count"]
        views_col = next((c for c in candidate_view_cols if c in df_all.columns), None)
        total_views = pd.to_numeric(df_all[views_col], errors="coerce").fillna(0).sum() if views_col else 0

        inter_cols = [c for c in ["likeCount","replyCount","retweetCount","quoteCount","bookmarkCount"] if c in df_all.columns]
        total_interacciones = (
            df_all[inter_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum()
            if inter_cols else 0
        )

        st.markdown(f"""
        <div style="text-align: center; padding: 12px 0;">
            <div style="font-size: 1.2em; font-weight: 700; color: #16a34a;">
                📈 Alcance Total: {int(total_views):,} visualizaciones
            </div>
            <div style="font-size: 1.2em; font-weight: 700; color: #2563eb;">
                💬 Interacciones Totales: {int(total_interacciones):,}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Previews (con fotos de perfil)
        if not df_replies.empty:
            st.write("### Algunas Respuestas")
            st.data_editor(
                df_replies.head(5),
                column_config={
                    "author/profilePicture": st.column_config.ImageColumn("Foto"),
                    "url": st.column_config.LinkColumn("Tweet"),
                    "author/userName": st.column_config.TextColumn("Usuario"),
                    "text": st.column_config.TextColumn("Contenido"),
                },
                hide_index=True,
                use_container_width=True
            )
        if not df_quotes.empty:
            st.write("### Algunas Citas")
            st.data_editor(
                df_quotes.head(5),
                column_config={
                    "author/profilePicture": st.column_config.ImageColumn("Foto"),
                    "url": st.column_config.LinkColumn("Tweet"),
                    "author/userName": st.column_config.TextColumn("Usuario"),
                    "text": st.column_config.TextColumn("Contenido"),
                },
                hide_index=True,
                use_container_width=True
            )

        # ---------- IA ----------
        st.subheader("🤖 Análisis con Gemini")
        analizar = st.button("Analizar Conversación con IA")
        if analizar:
            if not model:
                st.error("Configura `GEMINI_API_KEY` (env/secrets) para usar IA.")
            else:
                df_todos = df_all.copy()
                if df_todos.empty:
                    st.warning("No hay tweets para analizar.")
                else:
                    # Temas principales
                    with st.spinner("Extrayendo temas principales…"):
                        textos_todos = df_todos["text"].dropna().astype(str).tolist() if "text" in df_todos.columns else []
                        resultados = extraer_temas_con_ia(model, textos_todos, "mixto", contexto, num_temas=5)
                        st.markdown("### Temas principales detectados")
                        st.write(resultados)

                    # Clasificación de sentimientos (paralela)
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    resultados_sent = ["NEUTRO"] * len(df_todos)
                    validos = [(i, t) for i, t in enumerate(df_todos.get("text", pd.Series([], dtype=str))) if pd.notna(t) and str(t).strip()]
                    if validos:
                        progress = st.progress(0)
                        total = len(validos)
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            futures = {executor.submit(clasificar_tweet, model, t, contexto): i for i, t in validos}
                            for done, f in enumerate(as_completed(futures), 1):
                                idx = futures[f]
                                try:
                                    resultados_sent[idx] = f.result() or "NEUTRO"
                                except Exception:
                                    resultados_sent[idx] = "NEUTRO"
                                progress.progress(done/total)
                        st.success("✅ Clasificación completada.")
                        df_todos["sentimiento"] = resultados_sent

                    # Top 10 por vistas
                    st.markdown("---")
                    st.subheader("🔥 Top 10 Tweets Más Vistos")
                    if "viewCount" in df_todos.columns:
                        df_views = df_todos.dropna(subset=["viewCount"])
                        if not df_views.empty:
                            top_10 = df_views.sort_values("viewCount", ascending=False).head(10).copy()
                            top_10["viewCount"] = top_10["viewCount"].astype(int)
                            st.dataframe(
                                top_10,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "author/profilePicture": st.column_config.ImageColumn("Foto"),
                                    "url": st.column_config.LinkColumn("URL"),
                                    "viewCount": st.column_config.NumberColumn("Vistas", format="%d"),
                                    "createdAt": st.column_config.DatetimeColumn("Fecha"),
                                    "author/userName": st.column_config.TextColumn("Usuario"),
                                    "author/followers": st.column_config.NumberColumn("Seguidores", format="%d"),
                                    "likeCount": st.column_config.NumberColumn("Likes", format="%d"),
                                    "replyCount": st.column_config.NumberColumn("Respuestas", format="%d"),
                                    "retweetCount": st.column_config.NumberColumn("Retweets", format="%d"),
                                    "quoteCount": st.column_config.NumberColumn("Citas", format="%d"),
                                    "bookmarkCount": st.column_config.NumberColumn("Guardados", format="%d"),
                                    "text": st.column_config.TextColumn("Contenido"),
                                    "sentimiento": st.column_config.TextColumn("Sentimiento"),
                                    "tipo": st.column_config.TextColumn("Tipo"),
                                },
                            )
                    else:
                        st.info("No se encontró la columna `viewCount`.")

                    # Top 10 usuarios por seguidores
                    st.markdown("---")
                    st.subheader("👑 Top 10 Usuarios con Más Seguidores")
                    if {"author/followers","author/userName"}.issubset(df_todos.columns):
                        df_users = df_todos.dropna(subset=["author/followers", "author/userName"])
                        if not df_users.empty:
                            top_users = (
                                df_users.groupby("author/userName")
                                .agg(**{
                                    "author/followers": pd.NamedAgg(column="author/followers", aggfunc="max"),
                                    "author/profilePicture": pd.NamedAgg(column="author/profilePicture", aggfunc="first"),
                                })
                                .reset_index()
                                .sort_values("author/followers", ascending=False)
                                .head(10)
                            )
                            st.dataframe(
                                top_users,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "author/profilePicture": st.column_config.ImageColumn("Foto"),
                                    "author/userName": st.column_config.TextColumn("Usuario"),
                                    "author/followers": st.column_config.NumberColumn("Seguidores", format="%d"),
                                },
                            )

                    # Distribución de sentimientos
                    st.subheader("📊 Distribución de Sentimientos")
                    if "sentimiento" in df_todos.columns and not df_todos["sentimiento"].dropna().empty:
                        counts = df_todos["sentimiento"].value_counts().reset_index()
                        counts.columns = ["Sentimiento","Cantidad"]
                        counts["Porcentaje"] = counts["Cantidad"] / counts["Cantidad"].sum() * 100
                        st.dataframe(
                            counts.assign(Porcentaje=counts["Porcentaje"].round(2)),
                            hide_index=True,
                            use_container_width=True
                        )
                        fig = px.pie(counts, values="Cantidad", names="Sentimiento", title="Distribución de Sentimientos")
                        fig.update_traces(textposition="inside", textinfo="percent+label")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Aún no hay sentimientos clasificados.")

                    # Evolución temporal
                    if "createdAt" in df_todos.columns:
                        df_time = df_todos.copy()
                        df_time["createdAt"] = pd.to_datetime(df_time["createdAt"], errors="coerce", utc=True)
                        df_time = df_time.dropna(subset=["createdAt"])
                        if not df_time.empty:
                            min_date = df_time["createdAt"].min().date()
                            max_date = df_time["createdAt"].max().date()
                            date_range_days = (max_date - min_date).days
                            if date_range_days <= 3:
                                df_time["time_bucket"] = df_time["createdAt"].dt.strftime("%Y-%m-%d %H:00")
                                xaxis_label = "Hora"
                            elif date_range_days <= 150:
                                df_time["time_bucket"] = df_time["createdAt"].dt.date
                                xaxis_label = "Fecha"
                            else:
                                df_time["time_bucket"] = df_time["createdAt"].dt.to_period("M").astype(str)
                                xaxis_label = "Mes"

                            st.markdown("---")
                            st.subheader("📈 Evolución de Tweets en el Tiempo")
                            timeline = df_time.groupby("time_bucket").size().reset_index(name="Cantidad de Tweets")
                            fig_tl = px.line(timeline, x="time_bucket", y="Cantidad de Tweets",
                                             title=f"Cantidad de Tweets por {xaxis_label}", markers=True)
                            fig_tl.update_layout(xaxis_title=xaxis_label, yaxis_title="Número de Tweets")
                            st.plotly_chart(fig_tl, use_container_width=True)



    else:
        if url_input and not ejecutar:
            st.info("Pegá la URL y apretá **🚀 Ejecutar** para descargar los datos.")

# ---------- Entrada ----------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = True

if st.session_state["logged_in"]:
    main_app()


