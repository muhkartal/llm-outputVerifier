import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from PIL import Image
import base64
import json
from datetime import datetime, timedelta
import os
import io
import uuid
import hashlib
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from streamlit_option_menu import option_menu
from streamlit_extras.grid import grid
from streamlit_extras.chart_container import chart_container
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stateful_button import button
from streamlit_lottie import st_lottie
from fpdf import FPDF
import altair as alt
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')
import yaml
from yaml.loader import SafeLoader
import extra_streamlit_components as stx
import streamlit_toggle as toggle
import streamlit_shadcn_ui as ui
from streamlit_extras.stylable_container import stylable_container
from streamlit_card import card

import markdown
import httpx
from contextlib import contextmanager
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"
VERSION = "3.5.0"
CACHE_TTL = 3600  # Cache time to live in seconds
DEFAULT_MODEL = "Standard"
AVAILABLE_MODELS = ["Standard", "Advanced", "Expert"]
MAX_HISTORY_ITEMS = 100
MAX_BATCH_SIZE = 10

st.set_page_config(
    page_title="Reasoning Verifier ",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://reasoningverifier.ai/help',
        'Report a bug': 'https://reasoningverifier.ai/bug',
        'About': 'Reasoning Verifier v4.0.0 - Advanced validation engine for chain-of-thought reasoning in AI systems.'
    }
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "results" not in st.session_state:
    st.session_state.results = None
if "question" not in st.session_state:
    st.session_state.question = ""
if "reasoning" not in st.session_state:
    st.session_state.reasoning = ""
if "model" not in st.session_state:
    st.session_state.model = DEFAULT_MODEL
if "history" not in st.session_state:
    st.session_state.history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = set()
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "analyzer"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = True
if "user_name" not in st.session_state:
    st.session_state.user_name = "Demo User"
if "user_role" not in st.session_state:
    st.session_state.user_role = "Admin"
if "batch_jobs" not in st.session_state:
    st.session_state.batch_jobs = []
if "comparison_items" not in st.session_state:
    st.session_state.comparison_items = []
if "settings" not in st.session_state:
    st.session_state.settings = {
        "api_key": "demo_key_123456",
        "default_model": DEFAULT_MODEL,
        "auto_save": True,
        "notifications_enabled": True,
        "max_history": MAX_HISTORY_ITEMS,
        "theme": "dark",
        "accessibility": {
            "high_contrast": False,
            "large_text": False,
            "screen_reader_optimized": False
        }
    }


VERSION = "4.0.0"
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
CACHE_TTL = 3600
MAX_HISTORY_ITEMS = 50
SESSION_ID = str(uuid.uuid4())
DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() in ("true", "1", "t")
ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "True").lower() in ("true", "1", "t")
DEFAULT_THEME = "dark"

CONFIDENCE_THRESHOLDS = {
    "high": 0.85,
    "medium": 0.7,
    "low": 0.5
}

Path("logs").mkdir(exist_ok=True)
Path("exports").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)
Path("cache").mkdir(exist_ok=True)

def init_session_state():
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.theme = DEFAULT_THEME
        st.session_state.show_results = False
        st.session_state.results = None
        st.session_state.question = ""
        st.session_state.reasoning = ""
        st.session_state.history = []
        st.session_state.favorites = []
        st.session_state.model = DEFAULT_MODEL
        st.session_state.confidence_threshold = CONFIDENCE_THRESHOLDS["medium"]
        st.session_state.analytics_data = []
        st.session_state.notifications = []
        st.session_state.current_page = "dashboard"
        st.session_state.comparison_items = []
        st.session_state.batch_items = []
        st.session_state.filters = {
            "date_range": (datetime.now() - timedelta(days=30), datetime.now()),
            "min_confidence": 0.0,
            "max_hallucinations": 100,
            "search_term": ""
        }
        st.session_state.settings = {
            "notifications_enabled": True,
            "auto_save": True,
            "analytics_enabled": ANALYTICS_ENABLED,
            "theme": DEFAULT_THEME,
            "default_model": DEFAULT_MODEL,
            "export_format": "json",
            "max_history": MAX_HISTORY_ITEMS
        }
        st.session_state.tour_completed = False
        st.session_state.api_key = ""
        st.session_state.authenticated = DEMO_MODE
        st.session_state.user_role = "admin" if DEMO_MODE else "viewer"
        st.session_state.user_name = "Demo User" if DEMO_MODE else ""
        st.session_state.user_email = "demo@example.com" if DEMO_MODE else ""
        st.session_state.last_activity = datetime.now()
        st.session_state.sidebar_state = "expanded"



def load_auth_config():
    """Load authentication configuration."""
    if os.path.exists("config/auth.yaml"):
        with open("config/auth.yaml") as file:
            return yaml.load(file, Loader=SafeLoader)
    return {
        "credentials": {
            "usernames": {
                "admin": {
                    "name": "Admin User",
                    "password": stauth.Hasher(["admin"]).generate()[0],
                    "email": "admin@example.com",
                    "role": "admin"
                },
                "user": {
                    "name": "Regular User",
                    "password": stauth.Hasher(["password"]).generate()[0],
                    "email": "user@example.com",
                    "role": "viewer"
                }
            }
        },
        "cookie": {
            "name": "reasoning_verifier_auth",
            "key": "some_signature_key",
            "expiry_days": 30
        }
    }

def authenticate_user():
    """Authenticate user and set session state."""
    if st.session_state.authenticated:
        return True

    config = load_auth_config()
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"]
    )

    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status:
        st.session_state.authenticated = True
        st.session_state.user_name = name
        st.session_state.user_id = username
        st.session_state.user_role = config["credentials"]["usernames"][username]["role"]
        st.session_state.user_email = config["credentials"]["usernames"][username]["email"]
        return True
    elif authentication_status is False:
        st.error("Username/password is incorrect")

    return False

@contextmanager
def st_capture(output_func):
    """Capture stdout to a streamlit component."""
    with io.StringIO() as stdout, contextlib.redirect_stdout(stdout):
        old_write = stdout.write
        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        stdout.write = new_write
        yield

@st.cache_data(ttl=CACHE_TTL)
def load_example_data() -> Dict[str, Any]:
    """Load example data for demonstration purposes."""
    return {
        "simple": {
            "question": "What is the optimal resource allocation for the project?",
            "reasoning": """First, we need to identify the key variables in this problem. Given the project timeline of 18 months and resource allocation of $420,000, we need to determine the optimal distribution.
For resource utilization, we'll apply the standard efficiency formula: E = (Output / Input) × 100%. This gives us an efficiency rating of 84% for the current allocation model.
Based on historical data, we should assume a quarterly increase of approximately 3.2% in resource requirements.
Therefore, the optimal resource distribution would require $526,850 over the 18-month period to achieve the project objectives."""
        },
        "complex": {
            "question": "How will climate change affect agricultural yields in the Midwest by 2050?",
            "reasoning": """To analyze this question, I need to consider multiple climate models and their projections for the Midwest region.
The IPCC AR6 report indicates temperature increases of 1.5-3.5°C in the Midwest by 2050 under moderate emissions scenarios.
Higher temperatures will extend growing seasons but also increase heat stress on crops like corn and soybeans.
Precipitation patterns are projected to change, with more intense rainfall events but also longer dry periods.
Studies from the USDA suggest corn yields could decrease by 10-30% without adaptation strategies.
Soybean yields may be more resilient, with projected decreases of 5-15%.
Adaptation strategies like drought-resistant varieties and changed planting dates could mitigate some losses.
Therefore, without adaptation, Midwest agricultural yields will likely decrease by 10-25% by 2050, but with adaptation, the impact could be limited to 5-15% reductions or even maintain current productivity in some areas."""
        },
        "technical": {
            "question": "What are the security implications of using WebAssembly for browser-based cryptography?",
            "reasoning": """WebAssembly (Wasm) offers several advantages for implementing cryptography in browsers. First, it provides near-native performance, which is crucial for computationally intensive cryptographic operations.
Wasm's sandboxed execution model isolates the code from the rest of the browser, providing a security boundary that can help prevent certain types of attacks.
The deterministic execution of Wasm ensures consistent cryptographic operations across different browsers and platforms, which is essential for cryptographic algorithms.
However, Wasm doesn't inherently protect against side-channel attacks. Timing attacks remain possible as Wasm execution time can leak information about secret keys.
Additionally, Wasm modules can be reverse-engineered, potentially exposing cryptographic implementations and any weaknesses they might contain.
Browser-based cryptography, regardless of implementation, is vulnerable to client-side tampering. An attacker with control over the browser environment could modify the Wasm module or intercept calls.
For truly sensitive operations, server-side cryptography remains preferable, as the private keys never leave a controlled environment.
WebAssembly does offer the ability to implement more complex cryptographic algorithms efficiently in browsers, enabling better client-side encryption than was previously practical.
In conclusion, WebAssembly improves the performance and feasibility of browser-based cryptography but doesn't fundamentally solve the security challenges inherent to client-side cryptographic implementations."""
        }
    }

@st.cache_data(ttl=CACHE_TTL)
def get_logo_base64() -> str:
    """Return the base64 encoded logo."""
    return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIwIDJMMzUgMTJMMzUgMjhMMjAgMzhMNSAyOEw1IDEyTDIwIDJaIiBmaWxsPSJ1cmwoI3BhaW50MF9saW5lYXIpIi8+CjxwYXRoIGQ9Ik0yNyAxMy41TDE4IDIyLjVMMTMgMTcuNUkiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+CjxkZWZzPgo8bGluZWFyR3JhZGllbnQgaWQ9InBhaW50MF9saW5lYXIiIHgxPSI1IiB5MT0iMjAiIHgyPSIzNSIgeTI9IjIwIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+CjxzdG9wIHN0b3AtY29sb3I9IiM2MzY2RjEiLz4KPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjN0MzQUVEIi8+CjwvbGluZWFyR3JhZGllbnQ+CjwvZGVmcz4KPC9zdmc+Cg=="

@st.cache_data(ttl=CACHE_TTL)
def get_lottie_animation(animation_type: str) -> Dict[str, Any]:
    """Return a Lottie animation JSON for the specified type."""
    animations = {
        "loading": {
            "v": "5.7.4",
            "fr": 30,
            "ip": 0,
            "op": 60,
            "w": 100,
            "h": 100,
            "nm": "Loading",
            "ddd": 0,
            "assets": [],
            "layers": [
                {
                    "ddd": 0,
                    "ind": 1,
                    "ty": 4,
                    "nm": "Circle 1",
                    "sr": 1,
                    "ks": {
                        "o": {"a": 0, "k": 100, "ix": 11},
                        "r": {"a": 0, "k": 0, "ix": 10},
                        "p": {"a": 0, "k": [50, 50, 0], "ix": 2},
                        "a": {"a": 0, "k": [0, 0, 0], "ix": 1},
                        "s": {"a": 0, "k": [100, 100, 100], "ix": 6}
                    },
                    "ao": 0,
                    "shapes": [
                        {
                            "ty": "gr",
                            "it": [
                                {
                                    "d": 1,
                                    "ty": "el",
                                    "s": {"a": 0, "k": [40, 40], "ix": 2},
                                    "p": {"a": 0, "k": [0, 0], "ix": 3},
                                    "nm": "Ellipse Path 1",
                                    "mn": "ADBE Vector Shape - Ellipse",
                                    "hd": False
                                },
                                {
                                    "ty": "st",
                                    "c": {"a": 0, "k": [0.388, 0.4, 0.945, 1], "ix": 3},
                                    "o": {"a": 0, "k": 100, "ix": 4},
                                    "w": {"a": 0, "k": 6, "ix": 5},
                                    "lc": 2,
                                    "lj": 1,
                                    "ml": 4,
                                    "bm": 0,
                                    "nm": "Stroke 1",
                                    "mn": "ADBE Vector Graphic - Stroke",
                                    "hd": False
                                },
                                {
                                    "ty": "tr",
                                    "p": {"a": 0, "k": [0, 0], "ix": 2},
                                    "a": {"a": 0, "k": [0, 0], "ix": 1},
                                    "s": {"a": 0, "k": [100, 100], "ix": 3},
                                    "r": {"a": 0, "k": 0, "ix": 6},
                                    "o": {"a": 0, "k": 100, "ix": 7},
                                    "sk": {"a": 0, "k": 0, "ix": 4},
                                    "sa": {"a": 0, "k": 0, "ix": 5},
                                    "nm": "Transform"
                                }
                            ],
                            "nm": "Ellipse 1",
                            "np": 3,
                            "cix": 2,
                            "bm": 0,
                            "ix": 1,
                            "mn": "ADBE Vector Group",
                            "hd": False
                        },
                        {
                            "ty": "tm",
                            "s": {
                                "a": 1,
                                "k": [
                                    {
                                        "i": {"x": [0.667], "y": [1]},
                                        "o": {"x": [0.333], "y": [0]},
                                        "t": 0,
                                        "s": [0]
                                    },
                                    {"t": 60, "s": [100]}
                                ],
                                "ix": 1
                            },
                            "e": {
                                "a": 1,
                                "k": [
                                    {
                                        "i": {"x": [0.667], "y": [1]},
                                        "o": {"x": [0.333], "y": [0]},
                                        "t": 15,
                                        "s": [0]
                                    },
                                    {"t": 75, "s": [100]}
                                ],
                                "ix": 2
                            },
                            "o": {
                                "a": 1,
                                "k": [
                                    {
                                        "i": {"x": [0.667], "y": [1]},
                                        "o": {"x": [0.333], "y": [0]},
                                        "t": 0,
                                        "s": [0]
                                    },
                                    {"t": 60, "s": [360]}
                                ],
                                "ix": 3
                            },
                            "m": 1,
                            "ix": 2,
                            "nm": "Trim Paths 1",
                            "mn": "ADBE Vector Filter - Trim",
                            "hd": False
                        }
                    ],
                    "ip": 0,
                    "op": 300,
                    "st": 0,
                    "bm": 0
                }
            ],
            "markers": []
        },
        "success": {
            "v": "5.7.4",
            "fr": 30,
            "ip": 0,
            "op": 60,
            "w": 100,
            "h": 100,
            "nm": "Success",
            "ddd": 0,
            "assets": [],
            "layers": [
                {
                    "ddd": 0,
                    "ind": 1,
                    "ty": 4,
                    "nm": "Check",
                    "sr": 1,
                    "ks": {
                        "o": {"a": 0, "k": 100, "ix": 11},
                        "r": {"a": 0, "k": 0, "ix": 10},
                        "p": {"a": 0, "k": [50, 50, 0], "ix": 2},
                        "a": {"a": 0, "k": [0, 0, 0], "ix": 1},
                        "s": {"a": 0, "k": [100, 100, 100], "ix": 6}
                    },
                    "ao": 0,
                    "shapes": [
                        {
                            "ty": "gr",
                            "it": [
                                {
                                    "ind": 0,
                                    "ty": "sh",
                                    "ix": 1,
                                    "ks": {
                                        "a": 0,
                                        "k": {
                                            "i": [[0, 0], [0, 0], [0, 0]],
                                            "o": [[0, 0], [0, 0], [0, 0]],
                                            "v": [[-15, 2], [-5, 12], [15, -8]],
                                            "c": False
                                        },
                                        "ix": 2
                                    },
                                    "nm": "Path 1",
                                    "mn": "ADBE Vector Shape - Group",
                                    "hd": False
                                },
                                {
                                    "ty": "st",
                                    "c": {"a": 0, "k": [0.063, 0.722, 0.506, 1], "ix": 3},
                                    "o": {"a": 0, "k": 100, "ix": 4},
                                    "w": {"a": 0, "k": 6, "ix": 5},
                                    "lc": 2,
                                    "lj": 2,
                                    "bm": 0,
                                    "nm": "Stroke 1",
                                    "mn": "ADBE Vector Graphic - Stroke",
                                    "hd": False
                                },
                                {
                                    "ty": "tr",
                                    "p": {"a": 0, "k": [0, 0], "ix": 2},
                                    "a": {"a": 0, "k": [0, 0], "ix": 1},
                                    "s": {"a": 0, "k": [100, 100], "ix": 3},
                                    "r": {"a": 0, "k": 0, "ix": 6},
                                    "o": {"a": 0, "k": 100, "ix": 7},
                                    "sk": {"a": 0, "k": 0, "ix": 4},
                                    "sa": {"a": 0, "k": 0, "ix": 5},
                                    "nm": "Transform"
                                }
                            ],
                            "nm": "Shape 1",
                            "np": 3,
                            "cix": 2,
                            "bm": 0,
                            "ix": 1,
                            "mn": "ADBE Vector Group",
                            "hd": False
                        },
                        {
                            "ty": "tm",
                            "s": {"a": 0, "k": 0, "ix": 1},
                            "e": {
                                "a": 1,
                                "k": [
                                    {
                                        "i": {"x": [0.667], "y": [1]},
                                        "o": {"x": [0.333], "y": [0]},
                                        "t": 10,
                                        "s": [0]
                                    },
                                    {"t": 30, "s": [100]}
                                ],
                                "ix": 2
                            },
                            "o": {"a": 0, "k": 0, "ix": 3},
                            "m": 1,
                            "ix": 2,
                            "nm": "Trim Paths 1",
                            "mn": "ADBE Vector Filter - Trim",
                            "hd": False
                        }
                    ],
                    "ip": 0,
                    "op": 300,
                    "st": 0,
                    "bm": 0
                },
                {
                    "ddd": 0,
                    "ind": 2,
                    "ty": 4,
                    "nm": "Circle",
                    "sr": 1,
                    "ks": {
                        "o": {"a": 0, "k": 100, "ix": 11},
                        "r": {"a": 0, "k": 0, "ix": 10},
                        "p": {"a": 0, "k": [50, 50, 0], "ix": 2},
                        "a": {"a": 0, "k": [0, 0, 0], "ix": 1},
                        "s": {
                            "a": 1,
                            "k": [
                                {
                                    "i": {"x": [0.667, 0.667, 0.667], "y": [1, 1, 1]},
                                    "o": {"x": [0.333, 0.333, 0.333], "y": [0, 0, 0]},
                                    "t": 0,
                                    "s": [0, 0, 100]
                                },
                                {"t": 15, "s": [100, 100, 100]}
                            ],
                            "ix": 6
                        }
                    },
                    "ao": 0,
                    "shapes": [
                        {
                            "ty": "gr",
                            "it": [
                                {
                                    "d": 1,
                                    "ty": "el",
                                    "s": {"a": 0, "k": [40, 40], "ix": 2},
                                    "p": {"a": 0, "k": [0, 0], "ix": 3},
                                    "nm": "Ellipse Path 1",
                                    "mn": "ADBE Vector Shape - Ellipse",
                                    "hd": False
                                },
                                {
                                    "ty": "fl",
                                    "c": {"a": 0, "k": [0.063, 0.722, 0.506, 0.2], "ix": 4},
                                    "o": {"a": 0, "k": 100, "ix": 5},
                                    "r": 1,
                                    "bm": 0,
                                    "nm": "Fill 1",
                                    "mn": "ADBE Vector Graphic - Fill",
                                    "hd": False
                                },
                                {
                                    "ty": "tr",
                                    "p": {"a": 0, "k": [0, 0], "ix": 2},
                                    "a": {"a": 0, "k": [0, 0], "ix": 1},
                                    "s": {"a": 0, "k": [100, 100], "ix": 3},
                                    "r": {"a": 0, "k": 0, "ix": 6},
                                    "o": {"a": 0, "k": 100, "ix": 7},
                                    "sk": {"a": 0, "k": 0, "ix": 4},
                                    "sa": {"a": 0, "k": 0, "ix": 5},
                                    "nm": "Transform"
                                }
                            ],
                            "nm": "Ellipse 1",
                            "np": 3,
                            "cix": 2,
                            "bm": 0,
                            "ix": 1,
                            "mn": "ADBE Vector Group",
                            "hd": False
                        }
                    ],
                    "ip": 0,
                    "op": 300,
                    "st": 0,
                    "bm": 0
                }
            ],
            "markers": []
        }
    }

    return animations.get(animation_type, animations["loading"])

def local_css() -> None:
    """Apply custom CSS styling to the application."""
    try:
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback if file doesn't exist
        st.markdown(get_css(), unsafe_allow_html=True)

def get_css() -> str:
    """Return the CSS styling as a string."""
    return """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --primary: #6366F1;
        --primary-light: #818CF8;
        --primary-dark: #4F46E5;
        --secondary: #2D3748;
        --accent: #7C3AED;

        --bg-dark: #111827;
        --bg-card: #1E293B;
        --bg-card-hover: #1E293B;
        --bg-input: #1A2133;

        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-tertiary: #94A3B8;

        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --info: #3B82F6;

        --border-color: #2D3748;
        --divider: rgba(203, 213, 225, 0.1);

        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.5);

        --card-radius: 12px;
        --btn-radius: 8px;
        --badge-radius: 20px;
    }

    [data-theme="light"] {
        --primary: #4F46E5;
        --primary-light: #6366F1;
        --primary-dark: #4338CA;
        --secondary: #64748B;
        --accent: #7C3AED;

        --bg-dark: #F8FAFC;
        --bg-card: #FFFFFF;
        --bg-card-hover: #F1F5F9;
        --bg-input: #F8FAFC;

        --text-primary: #1E293B;
        --text-secondary: #334155;
        --text-tertiary: #64748B;

        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --info: #3B82F6;

        --border-color: #E2E8F0;
        --divider: rgba(100, 116, 139, 0.1);

        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
    }

    [data-theme="high_contrast"] {
        --primary: #FFFFFF;
        --primary-light: #FFFFFF;
        --primary-dark: #FFFFFF;
        --secondary: #FFFFFF;
        --accent: #FFFFFF;

        --bg-dark: #000000;
        --bg-card: #121212;
        --bg-card-hover: #1A1A1A;
        --bg-input: #1A1A1A;

        --text-primary: #FFFFFF;
        --text-secondary: #EEEEEE;
        --text-tertiary: #CCCCCC;

        --success: #00FF00;
        --warning: #FFFF00;
        --danger: #FF0000;
        --info: #00FFFF;

        --border-color: #FFFFFF;
        --divider: rgba(255, 255, 255, 0.2);

        --shadow-sm: 0 2px 4px rgba(255, 255, 255, 0.2);
        --shadow-md: 0 4px 8px rgba(255, 255, 255, 0.2);
        --shadow-lg: 0 10px 15px rgba(255, 255, 255, 0.2);
    }

    body {
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }

    .stApp {
        background-color: var(--bg-dark);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    a {
        color: var(--primary-light);
        text-decoration: none;
        transition: color 0.2s ease;
    }

    a:hover {
        color: var(--primary);
        text-decoration: none;
    }

    p {
        color: var(--text-secondary);
    }

    .app-container {
        padding: 0 1rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--divider);
    }

    .header-left {
        display: flex;
        flex-direction: column;
    }

    .logo-container {
        display: flex;
        align-items: center;
    }

    .logo-icon {
        width: 40px;
        height: 40px;
        margin-right: 12px;
        filter: drop-shadow(0 0 6px rgba(99, 102, 241, 0.4));
    }

    .app-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
        margin: 0;
        padding: 0;
        letter-spacing: -0.03em;
    }

    .-badge {
        background: linear-gradient(90deg, var(--primary-dark) 0%, var(--accent) 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: 12px;
        box-shadow: 0 2px 4px rgba(124, 58, 237, 0.3);
    }

    .header-subtitle {
        color: var(--text-tertiary);
        font-size: 1rem;
        font-weight: 400;
        margin-top: 8px;
        max-width: 650px;
    }

    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--divider);
        display: flex;
        align-items: center;
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        margin-right: 8px;
        border-radius: 6px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .section-icon svg {
        width: 16px;
        height: 16px;
    }

    .content-card {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .content-card:hover {
        box-shadow: var(--shadow-md);
        border-color: rgba(99, 102, 241, 0.3);
    }

    .content-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
        opacity: 0.8;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, rgba(36, 46, 67, 0.8) 100%);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%);
        opacity: 0.8;
    }

    .metric-card.warning::before {
        background: linear-gradient(90deg, var(--warning) 0%, #FBBF24 100%);
    }

    .metric-card.danger::before {
        background: linear-gradient(90deg, var(--danger) 0%, #F87171 100%);
    }

    .metric-card.success::before {
        background: linear-gradient(90deg, var(--success) 0%, #34D399 100%);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-md);
    }

    .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    .metric-card.warning .metric-icon {
        background: linear-gradient(135deg, var(--warning) 0%, #FBBF24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    .metric-card.danger .metric-icon {
        background: linear-gradient(135deg, var(--danger) 0%, #F87171 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    .metric-card.success .metric-icon {
        background: linear-gradient(135deg, var(--success) 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.85rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Result steps */
    .result-step {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        padding: 1.25rem;
        margin-bottom: 1rem;
        position: relative;
        transition: all 0.2s ease;
    }

    .result-step::before {
        content: '';
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        width: 4px;
        border-radius: 4px 0 0 4px;
    }

    .result-step.grounded-step::before {
        background-color: var(--success);
    }

    .result-step.likely-step::before {
        background-color: var(--primary-light);
    }

    .result-step.warning-step::before {
        background-color: var(--warning);
    }

    .result-step.danger-step::before {
        background-color: var(--danger);
    }

    .result-step:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.2);
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: var(--badge-radius);
        font-weight: 600;
        font-size: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .status-grounded {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .status-likely {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-light);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    .status-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }

    .status-danger {
        background-color: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .confidence-badge {
        background-color: rgba(255, 255, 255, 0.05);
        color: var(--text-tertiary);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        margin-left: 8px;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Chart containers */
    .chart-container {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.2s ease;
    }

    .chart-container:hover {
        box-shadow: var(--shadow-md);
        border-color: rgba(99, 102, 241, 0.2);
    }

    /* Form elements */
    .stTextInput > div {
        background-color: var(--bg-dark) !important;
    }

    .stTextInput > div > div > input {
        background-color: var(--bg-input) !important;
        border-radius: var(--btn-radius) !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        border: 1px solid var(--border-color) !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    .stTextArea > div {
        background-color: var(--bg-dark) !important;
    }

    .stTextArea > div > div > textarea {
        background-color: var(--bg-input) !important;
        border-radius: var(--btn-radius) !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        border: 1px solid var(--border-color) !important;
        padding: 0.75rem 1rem !important;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.65rem 2rem !important;
        border-radius: var(--btn-radius) !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.25) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        box-shadow: 0 6px 10px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 4px rgba(99, 102, 241, 0.25) !important;
    }

    /* Expanders */
    div.stExpander {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }

    div.stExpander > div:first-child {
        background-color: var(--bg-card) !important;
        border-radius: var(--card-radius) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.2s ease !important;
    }

    div.stExpander > div:first-child:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
        box-shadow: var(--shadow-md) !important;
    }

    div.stExpander > details > summary {
        padding: 1rem 1.5rem !important;
        border-radius: var(--card-radius) !important;
    }

    div.stExpander > details > summary:hover {
        background-color: rgba(99, 102, 241, 0.05) !important;
    }

    div.stExpander > details summary p {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 1.05rem !important;
    }

    div.stExpander > details > summary > div {
        color: var(--text-primary) !important;
    }

    div.stExpander > details[open] div {
        padding: 1.5rem !important;
        border-top: 1px solid var(--border-color) !important;
        background-color: var(--bg-card) !important;
    }

    /* Insights container */
    .insights-container {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.05) 0%, rgba(124, 58, 237, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: var(--card-radius);
        padding: 1.75rem;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .insights-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%236366F1' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
        pointer-events: none;
    }

    .insights-header {
        font-weight: 700;
        color: var(--primary-light);
        display: flex;
        align-items: center;
        margin-bottom: 1.25rem;
        font-size: 1.1rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .insights-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.1);
    }

    .insights-item:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .insights-bullet {
        color: var(--primary-light);
        margin-right: 0.75rem;
        font-weight: bold;
        flex-shrink: 0;
    }

    /* Footer */
    .custom-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--divider);
        text-align: center;
        color: var(--text-tertiary);
        font-size: 0.85rem;
    }

    .footer-links {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 0.75rem;
    }

    .footer-link {
        color: var(--text-tertiary);
        text-decoration: none;
        transition: color 0.2s ease;
    }

    .footer-link:hover {
        color: var(--primary-light);
        text-decoration: none;
    }

    /* Lists */
    .benefits-list {
        list-style-type: none;
        padding: 0;
        margin: 0;
    }

    .benefits-list li {
        position: relative;
        padding-left: 1.75rem;
        margin-bottom: 0.75rem;
        color: var(--text-secondary);
    }

    .benefits-list li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.5rem;
        width: 0.75rem;
        height: 0.75rem;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    }

    /* Glass effect */
    .glass-effect {
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Process steps */
    .process-steps {
        counter-reset: step;
        list-style-type: none;
        padding: 0;
        margin: 0;
    }

    .process-steps li {
        position: relative;
        padding-left: 2.5rem;
        margin-bottom: 1rem;
        color: var(--text-secondary);
    }

    .process-steps li::before {
        counter-increment: step;
        content: counter(step);
        position: absolute;
        left: 0;
        top: 0;
        width: 1.75rem;
        height: 1.75rem;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Legend items */
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--divider);
    }

    .legend-item:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
    }

    .legend-icon {
        width: 1.5rem;
        height: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        margin-right: 0.75rem;
        flex-shrink: 0;
    }

    .legend-grounded {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .legend-likely {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-light);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    .legend-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }

    .legend-danger {
        background-color: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .legend-text {
        flex: 1;
    }

    .legend-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .legend-desc {
        color: var(--text-tertiary);
        font-size: 0.85rem;
    }

    /* Theme toggle */
    .theme-toggle {
        display: flex;
        align-items: center;
        margin-left: auto;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 20px;
        padding: 4px;
        border: 1px solid var(--border-color);
    }

    .theme-toggle button {
        background: transparent;
        border: none;
        color: var(--text-tertiary);
        padding: 6px 12px;
        border-radius: 16px;
        cursor: pointer;
        font-size: 0.8rem;
        transition: all 0.2s ease;
    }

    .theme-toggle button.active {
        background: rgba(99, 102, 241, 0.2);
        color: var(--primary-light);
    }

    /* Export options */
    .export-options {
        position: absolute;
        top: 1rem;
        right: 1rem;
        z-index: 10;
    }

    .export-dropdown {
        position: relative;
        display: inline-block;
    }

    .export-btn {
        background: rgba(30, 41, 59, 0.7);
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
        border-radius: var(--btn-radius);
        padding: 6px 12px;
        font-size: 0.8rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .export-content {
        display: none;
        position: absolute;
        right: 0;
        background: var(--bg-card);
        min-width: 160px;
        box-shadow: var(--shadow-md);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        z-index: 1;
    }

    .export-content a {
        color: var(--text-secondary);
        padding: 10px 16px;
        text-decoration: none;
        display: block;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }

    .export-content a:hover {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-light);
    }

    .export-dropdown:hover .export-content {
        display: block;
    }

    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltip-text {
        visibility: hidden;
        width: 200px;
        background-color: var(--bg-card);
        color: var(--text-secondary);
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-md);
    }

    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    /* Loading animation */
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }

    .loading-dot {
        width: 12px;
        height: 12px;
        margin: 0 6px;
        border-radius: 50%;
        background-color: var(--primary);
        animation: pulse 1.5s infinite ease-in-out;
    }

    .loading-dot:nth-child(2) {
        animation-delay: 0.2s;
        background-color: var(--primary-light);
    }

    .loading-dot:nth-child(3) {
        animation-delay: 0.4s;
        background-color: var(--accent);
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(0.8);
            opacity: 0.6;
        }
        50% {
            transform: scale(1.2);
            opacity: 1;
        }
    }

    /* History panel */
    .history-panel {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .history-item {
        padding: 0.75rem;
        border-radius: var(--btn-radius);
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }

    .history-item:hover {
        background-color: rgba(99, 102, 241, 0.05);
        border-color: rgba(99, 102, 241, 0.2);
    }

    .history-item.active {
        background-color: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.3);
    }

    .history-question {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
    }

    .history-date {
        color: var(--text-tertiary);
        font-size: 0.75rem;
    }

    /* Search container */
    .search-container {
        position: relative;
        margin-bottom: 1rem;
    }

    .search-icon {
        position: absolute;
        left: 12px;
        top: 50%;
        transform: translateY(-50%);
        color: var(--text-tertiary);
    }

    .search-input {
        width: 100%;
        padding: 0.6rem 1rem 0.6rem 2.5rem;
        border-radius: var(--btn-radius);
        border: 1px solid var(--border-color);
        background-color: var(--bg-input);
        color: var(--text-primary);
        font-size: 0.9rem;
    }

    .search-input:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }

    /* Dashboard cards */
    .dashboard-card {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }

    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-md);
        border-color: rgba(99, 102, 241, 0.3);
    }

    .dashboard-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .dashboard-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .dashboard-card-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        color: var(--primary);
    }

    /* Navigation menu */
    .nav-menu {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .nav-item {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        color: var(--text-secondary);
        text-decoration: none;
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
    }

    .nav-item:hover {
        background-color: rgba(99, 102, 241, 0.05);
        color: var(--primary-light);
    }

    .nav-item.active {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-light);
        border-left: 3px solid var(--primary);
    }

    .nav-icon {
        margin-right: 0.75rem;
        width: 20px;
        height: 20px;
    }

    /* User profile */
    .user-profile {
        display: flex;
        align-items: center;
        padding: 1rem;
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        margin-right: 0.75rem;
    }

    .user-info {
        flex: 1;
    }

    .user-name {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .user-role {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-bottom: 1.5rem;
    }

    .comparison-table th,
    .comparison-table td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }

    .comparison-table th {
        background-color: rgba(30, 41, 59, 0.5);
        color: var(--text-primary);
        font-weight: 600;
    }

    .comparison-table tr:last-child td {
        border-bottom: none;
    }

    .comparison-table tr:hover td {
        background-color: rgba(99, 102, 241, 0.05);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: var(--bg-card);
        border-radius: var(--card-radius) var(--card-radius) 0 0;
        padding: 0.5rem 0.5rem 0 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: var(--btn-radius) var(--btn-radius) 0 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--bg-dark) !important;
        color: var(--primary-light) !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background-color: var(--bg-dark);
        border-radius: 0 0 var(--card-radius) var(--card-radius);
        padding: 1rem;
    }

    /* Responsive styles */
    @media (max-width: 768px) {
        .app-header {
            flex-direction: column;
            align-items: flex-start;
        }

        .header-subtitle {
            max-width: 100%;
        }

        .theme-toggle {
            margin-left: 0;
            margin-top: 1rem;
        }
    }
    """

def apply_theme():
    """Apply the selected theme to the application."""
    theme = st.session_state.theme

    # Apply CSS based on theme
    if theme == "dark":
        st.markdown(get_dark_theme_css(), unsafe_allow_html=True)
    elif theme == "light":
        st.markdown(get_light_theme_css(), unsafe_allow_html=True)
    elif theme == "high_contrast":
        st.markdown(get_high_contrast_theme_css(), unsafe_allow_html=True)
    else:
        st.markdown(get_dark_theme_css(), unsafe_allow_html=True)

def get_dark_theme_css() -> str:
    """Return the CSS for dark theme."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --primary: #6366F1;
        --primary-light: #818CF8;
        --primary-dark: #4F46E5;
        --secondary: #2D3748;
        --accent: #7C3AED;

        --bg-dark: #0F172A;
        --bg-card: #1E293B;
        --bg-card-hover: #1E293B;
        --bg-input: #1A2133;

        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-tertiary: #94A3B8;

        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --info: #3B82F6;

        --border-color: #2D3748;
        --divider: rgba(203, 213, 225, 0.1);

        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.5);

        --card-radius: 12px;
        --btn-radius: 8px;
        --badge-radius: 20px;
    }

    /* Main styles */
    body {
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }

    .stApp {
        background-color: var(--bg-dark);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    a {
        color: var(--primary-light);
        text-decoration: none;
        transition: color 0.2s ease;
    }

    a:hover {
        color: var(--primary);
        text-decoration: none;
    }

    p {
        color: var(--text-secondary);
    }

    /* App container */
    .app-container {
        padding: 0 1rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--divider);
    }

    .header-left {
        display: flex;
        flex-direction: column;
    }

    .logo-container {
        display: flex;
        align-items: center;
    }

    .logo-icon {
        width: 40px;
        height: 40px;
        margin-right: 12px;
        filter: drop-shadow(0 0 6px rgba(99, 102, 241, 0.4));
    }

    .app-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
        margin: 0;
        padding: 0;
        letter-spacing: -0.03em;
    }

    .-badge {
        background: linear-gradient(90deg, var(--primary-dark) 0%, var(--accent) 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: 12px;
        box-shadow: 0 2px 4px rgba(124, 58, 237, 0.3);
    }

    .header-subtitle {
        color: var(--text-tertiary);
        font-size: 1rem;
        font-weight: 400;
        margin-top: 8px;
        max-width: 650px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--divider);
        display: flex;
        align-items: center;
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        margin-right: 8px;
        border-radius: 6px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .section-icon svg {
        width: 16px;
        height: 16px;
    }

    /* Cards */
    .content-card {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .content-card:hover {
        box-shadow: var(--shadow-md);
        border-color: rgba(99, 102, 241, 0.3);
    }

    .content-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
        opacity: 0.8;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, var(--bg-card) 0%, rgba(36, 46, 67, 0.8) 100%);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%);
        opacity: 0.8;
    }

    .metric-card.warning::before {
        background: linear-gradient(90deg, var(--warning) 0%, #FBBF24 100%);
    }

    .metric-card.danger::before {
        background: linear-gradient(90deg, var(--danger) 0%, #F87171 100%);
    }

    .metric-card.success::before {
        background: linear-gradient(90deg, var(--success) 0%, #34D399 100%);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-md);
    }

    .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    .metric-card.warning .metric-icon {
        background: linear-gradient(135deg, var(--warning) 0%, #FBBF24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    .metric-card.danger .metric-icon {
        background: linear-gradient(135deg, var(--danger) 0%, #F87171 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    .metric-card.success .metric-icon {
        background: linear-gradient(135deg, var(--success) 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: var(--text-primary);
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.85rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Result steps */
    .result-step {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        padding: 1.25rem;
        margin-bottom: 1rem;
        position: relative;
        transition: all 0.2s ease;
    }

    .result-step::before {
        content: '';
        position: absolute;
        top: 0;
        bottom: 0;
        left: 0;
        width: 4px;
        border-radius: 4px 0 0 4px;
    }

    .result-step.grounded-step::before {
        background-color: var(--success);
    }

    .result-step.likely-step::before {
        background-color: var(--primary-light);
    }

    .result-step.warning-step::before {
        background-color: var(--warning);
    }

    .result-step.danger-step::before {
        background-color: var(--danger);
    }

    .result-step:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.2);
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: var(--badge-radius);
        font-weight: 600;
        font-size: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .status-grounded {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .status-likely {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-light);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    .status-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }

    .status-danger {
        background-color: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .confidence-badge {
        background-color: rgba(255, 255, 255, 0.05);
        color: var(--text-tertiary);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        margin-left: 8px;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Chart containers */
    .chart-container {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.2s ease;
    }

    .chart-container:hover {
        box-shadow: var(--shadow-md);
        border-color: rgba(99, 102, 241, 0.2);
    }

    /* Form elements */
    .stTextInput > div {
        background-color: var(--bg-dark) !important;
    }

    .stTextInput > div > div > input {
        background-color: var(--bg-input) !important;
        border-radius: var(--btn-radius) !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        border: 1px solid var(--border-color) !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    .stTextArea > div {
        background-color: var(--bg-dark) !important;
    }

    .stTextArea > div > div > textarea {
        background-color: var(--bg-input) !important;
        border-radius: var(--btn-radius) !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        border: 1px solid var(--border-color) !important;
        padding: 0.75rem 1rem !important;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.65rem 2rem !important;
        border-radius: var(--btn-radius) !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.25) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        box-shadow: 0 6px 10px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 4px rgba(99, 102, 241, 0.25) !important;
    }

    /* Expanders */
    div.stExpander {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }

    div.stExpander > div:first-child {
        background-color: var(--bg-card) !important;
        border-radius: var(--card-radius) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.2s ease !important;
    }

    div.stExpander > div:first-child:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
        box-shadow: var(--shadow-md) !important;
    }

    div.stExpander > details > summary {
        padding: 1rem 1.5rem !important;
        border-radius: var(--card-radius) !important;
    }

    div.stExpander > details > summary:hover {
        background-color: rgba(99, 102, 241, 0.05) !important;
    }

    div.stExpander > details summary p {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 1.05rem !important;
    }

    div.stExpander > details > summary > div {
        color: var(--text-primary) !important;
    }

    div.stExpander > details[open] div {
        padding: 1.5rem !important;
        border-top: 1px solid var(--border-color) !important;
        background-color: var(--bg-card) !important;
    }

    /* Insights container */
    .insights-container {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.05) 0%, rgba(124, 58, 237, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: var(--card-radius);
        padding: 1.75rem;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .insights-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%236366F1' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E");
        pointer-events: none;
    }

    .insights-header {
        font-weight: 700;
        color: var(--primary-light);
        display: flex;
        align-items: center;
        margin-bottom: 1.25rem;
        font-size: 1.1rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .insights-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.1);
    }

    .insights-item:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .insights-bullet {
        color: var(--primary-light);
        margin-right: 0.75rem;
        font-weight: bold;
        flex-shrink: 0;
    }

    /* Footer */
    .custom-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--divider);
        text-align: center;
        color: var(--text-tertiary);
        font-size: 0.85rem;
    }

    .footer-links {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 0.75rem;
    }

    .footer-link {
        color: var(--text-tertiary);
        text-decoration: none;
        transition: color 0.2s ease;
    }

    .footer-link:hover {
        color: var(--primary-light);
        text-decoration: none;
    }

    /* Lists */
    .benefits-list {
        list-style-type: none;
        padding: 0;
        margin: 0;
    }

    .benefits-list li {
        position: relative;
        padding-left: 1.75rem;
        margin-bottom: 0.75rem;
        color: var(--text-secondary);
    }

    .benefits-list li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.5rem;
        width: 0.75rem;
        height: 0.75rem;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    }

    /* Glass effect */
    .glass-effect {
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Process steps */
    .process-steps {
        counter-reset: step;
        list-style-type: none;
        padding: 0;
        margin: 0;
    }

    .process-steps li {
        position: relative;
        padding-left: 2.5rem;
        margin-bottom: 1rem;
        color: var(--text-secondary);
    }

    .process-steps li::before {
        counter-increment: step;
        content: counter(step);
        position: absolute;
        left: 0;
        top: 0;
        width: 1.75rem;
        height: 1.75rem;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Legend items */
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--divider);
    }

    .legend-item:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
    }

    .legend-icon {
        width: 1.5rem;
        height: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        margin-right: 0.75rem;
        flex-shrink: 0;
    }

    .legend-grounded {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .legend-likely {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-light);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    .legend-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }

    .legend-danger {
        background-color: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    .legend-text {
        flex: 1;
    }

    .legend-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .legend-desc {
        color: var(--text-tertiary);
        font-size: 0.85rem;
    }

    /* Theme toggle */
    .theme-toggle {
        display: flex;
        align-items: center;
        margin-left: auto;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 20px;
        padding: 4px;
        border: 1px solid var(--border-color);
    }

    .theme-toggle button {
        background: transparent;
        border: none;
        color: var(--text-tertiary);
        padding: 6px 12px;
        border-radius: 16px;
        cursor: pointer;
        font-size: 0.8rem;
        transition: all 0.2s ease;
    }

    .theme-toggle button.active {
        background: rgba(99, 102, 241, 0.2);
        color: var(--primary-light);
    }

    /* Export options */
    .export-options {
        position: absolute;
        top: 1rem;
        right: 1rem;
        z-index: 10;
    }

    .export-dropdown {
        position: relative;
        display: inline-block;
    }

    .export-btn {
        background: rgba(30, 41, 59, 0.7);
        color: var(--text-secondary);
        border: 1px solid var(--border-color);
        border-radius: var(--btn-radius);
        padding: 6px 12px;
        font-size: 0.8rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .export-content {
        display: none;
        position: absolute;
        right: 0;
        background: var(--bg-card);
        min-width: 160px;
        box-shadow: var(--shadow-md);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        z-index: 1;
    }

    .export-content a {
        color: var(--text-secondary);
        padding: 10px 16px;
        text-decoration: none;
        display: block;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }

    .export-content a:hover {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-light);
    }

    .export-dropdown:hover .export-content {
        display: block;
    }

    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltip-text {
        visibility: hidden;
        width: 200px;
        background-color: var(--bg-card);
        color: var(--text-secondary);
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-md);
    }

    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    /* Loading animation */
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }

    .loading-dot {
        width: 12px;
        height: 12px;
        margin: 0 6px;
        border-radius: 50%;
        background-color: var(--primary);
        animation: pulse 1.5s infinite ease-in-out;
    }

    .loading-dot:nth-child(2) {
        animation-delay: 0.2s;
        background-color: var(--primary-light);
    }

    .loading-dot:nth-child(3) {
        animation-delay: 0.4s;
        background-color: var(--accent);
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(0.8);
            opacity: 0.6;
        }
        50% {
            transform: scale(1.2);
            opacity: 1;
        }
    }

    /* History panel */
    .history-panel {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .history-item {
        padding: 0.75rem;
        border-radius: var(--btn-radius);
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }

    .history-item:hover {
        background-color: rgba(99, 102, 241, 0.05);
        border-color: rgba(99, 102, 241, 0.2);
    }

    .history-item.active {
        background-color: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.3);
    }

    .history-question {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
    }

    .history-date {
        color: var(--text-tertiary);
        font-size: 0.75rem;
    }

    /* Search container */
    .search-container {
        position: relative;
        margin-bottom: 1rem;
    }

    .search-icon {
        position: absolute;
        left: 12px;
        top: 50%;
        transform: translateY(-50%);
        color: var(--text-tertiary);
    }

    .search-input {
        width: 100%;
        padding: 0.6rem 1rem 0.6rem 2.5rem;
        border-radius: var(--btn-radius);
        border: 1px solid var(--border-color);
        background-color: var(--bg-input);
        color: var(--text-primary);
        font-size: 0.9rem;
    }

    .search-input:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }

    /* Dashboard cards */
    .dashboard-card {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }

    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-md);
        border-color: rgba(99, 102, 241, 0.3);
    }

    .dashboard-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .dashboard-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .dashboard-card-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        color: var(--primary);
    }

    /* Navigation menu */
    .nav-menu {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .nav-item {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        color: var(--text-secondary);
        text-decoration: none;
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
    }

    .nav-item:hover {
        background-color: rgba(99, 102, 241, 0.05);
        color: var(--primary-light);
    }

    .nav-item.active {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--primary-light);
        border-left: 3px solid var(--primary);
    }

    .nav-icon {
        margin-right: 0.75rem;
        width: 20px;
        height: 20px;
    }

    /* User profile */
    .user-profile {
        display: flex;
        align-items: center;
        padding: 1rem;
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        margin-right: 0.75rem;
    }

    .user-info {
        flex: 1;
    }

    .user-name {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .user-role {
        font-size: 0.75rem;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-bottom: 1.5rem;
    }

    .comparison-table th,
    .comparison-table td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }

    .comparison-table th {
        background-color: rgba(30, 41, 59, 0.5);
        color: var(--text-primary);
        font-weight: 600;
    }

    .comparison-table tr:last-child td {
        border-bottom: none;
    }

    .comparison-table tr:hover td {
        background-color: rgba(99, 102, 241, 0.05);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: var(--bg-card);
        border-radius: var(--card-radius) var(--card-radius) 0 0;
        padding: 0.5rem 0.5rem 0 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: var(--btn-radius) var(--btn-radius) 0 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--bg-dark) !important;
        color: var(--primary-light) !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background-color: var(--bg-dark);
        border-radius: 0 0 var(--card-radius) var(--card-radius);
        padding: 1rem;
    }

    /* Responsive styles */
    @media (max-width: 768px) {
        .app-header {
            flex-direction: column;
            align-items: flex-start;
        }

        .header-subtitle {
            max-width: 100%;
        }

        .theme-toggle {
            margin-left: 0;
            margin-top: 1rem;
        }
    }
    </style>
    """

def get_light_theme_css() -> str:
    """Return the CSS for light theme."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --primary: #4F46E5;
        --primary-light: #6366F1;
        --primary-dark: #4338CA;
        --secondary: #64748B;
        --accent: #7C3AED;

        --bg-light: #F8FAFC;
        --bg-card: #FFFFFF;
        --bg-card-hover: #F1F5F9;
        --bg-input: #F1F5F9;

        --text-primary: #1E293B;
        --text-secondary: #334155;
        --text-tertiary: #64748B;

        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --info: #3B82F6;

        --border-color: #E2E8F0;
        --divider: rgba(100, 116, 139, 0.1);

        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);

        --card-radius: 12px;
        --btn-radius: 8px;
        --badge-radius: 20px;
    }

    /* Main styles */
    body {
        background-color: var(--bg-light);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }

    .stApp {
        background-color: var(--bg-light);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    a {
        color: var(--primary);
        text-decoration: none;
        transition: color 0.2s ease;
    }

    a:hover {
        color: var(--primary-dark);
        text-decoration: none;
    }

    p {
        color: var(--text-secondary);
    }

    /* App container */
    .app-container {
        padding: 0 1rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border-color);
    }

    .header-left {
        display: flex;
        flex-direction: column;
    }

    .logo-container {
        display: flex;
        align-items: center;
    }

    .logo-icon {
        width: 40px;
        height: 40px;
        margin-right: 12px;
        filter: drop-shadow(0 0 6px rgba(79, 70, 229, 0.4));
    }

    .app-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
        margin: 0;
        padding: 0;
        letter-spacing: -0.03em;
    }

    .-badge {
        background: linear-gradient(90deg, var(--primary-dark) 0%, var(--accent) 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: 12px;
        box-shadow: 0 2px 4px rgba(124, 58, 237, 0.3);
    }

    .header-subtitle {
        color: var(--text-tertiary);
        font-size: 1rem;
        font-weight: 400;
        margin-top: 8px;
        max-width: 650px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--divider);
        display: flex;
        align-items: center;
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        margin-right: 8px;
        border-radius: 6px;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        border: 1px solid rgba(79, 70, 229, 0.3);
    }

    .section-icon svg {
        width: 16px;
        height: 16px;
    }

    /* Cards */
    .content-card {
        background-color: var(--bg-card);
        border-radius: var(--card-radius);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .content-card:hover {
        box-shadow: var(--shadow-md);
        border-color: rgba(79, 70, 229, 0.3);
    }

    .content-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
        opacity: 0.8;
    }

    /* And all other CSS styles continue... */
    </style>
    """

def get_high_contrast_theme_css() -> str:
    """Return the CSS for high contrast theme."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --primary: #FFFF00;
        --primary-light: #FFFF80;
        --primary-dark: #CCCC00;
        --secondary: #FFFFFF;
        --accent: #00FFFF;

        --bg-dark: #000000;
        --bg-card: #0A0A0A;
        --bg-card-hover: #1A1A1A;
        --bg-input: #0F0F0F;

        --text-primary: #FFFFFF;
        --text-secondary: #EEEEEE;
        --text-tertiary: #CCCCCC;

        --success: #00FF00;
        --warning: #FFFF00;
        --danger: #FF0000;
        --info: #00FFFF;

        --border-color: #333333;
        --divider: rgba(255, 255, 255, 0.2);

        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.5);
        --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.6);
        --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.7);

        --card-radius: 12px;
        --btn-radius: 8px;
        --badge-radius: 20px;
    }

    /* Main styles */
    body {
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }

    .stApp {
        background-color: var(--bg-dark);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }

    a {
        color: var(--primary);
        text-decoration: none;
        transition: color 0.2s ease;
    }

    a:hover {
        color: var(--primary-light);
        text-decoration: underline;
    }

    p {
        color: var(--text-secondary);
    }

    /* High contrast specific styles */
    .stButton > button {
        background: var(--primary) !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        font-weight: 700 !important;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid var(--border-color) !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border: 2px solid var(--primary) !important;
        box-shadow: 0 0 0 2px var(--primary) !important;
    }

    /* And all other high contrast CSS styles... */
    </style>
    """

@st.cache_data(ttl=CACHE_TTL)
def format_step(
    step: str,
    is_hallucination: bool,
    confidence: float,
    is_reliable: bool,
) -> str:
    """Format a reasoning step with appropriate styling based on its classification."""
    confidence_text = f"{confidence:.2f}"

    if is_hallucination:
        if is_reliable:
            step_class = "result-step danger-step"
            status_class = "status-badge status-danger"
            icon = "⚠️"
            status_text = "Hallucination"
        else:
            step_class = "result-step warning-step"
            status_class = "status-badge status-warning"
            icon = "⚠️"
            status_text = "Potential Hallucination"
    else:
        if is_reliable:
            step_class = "result-step grounded-step"
            status_class = "status-badge status-grounded"
            icon = "✓"
            status_text = "Grounded"
        else:
            step_class = "result-step likely-step"
            status_class = "status-badge status-likely"
            icon = "✓"
            status_text = "Likely Grounded"

    html = f"""
    <div class="{step_class}">
        <div class="{status_class}">
            {icon} {status_text}
            <span class="confidence-badge">{confidence_text}</span>
        </div>
        <div style="margin-top: 0.5rem; color: var(--text-secondary);">{step}</div>
    </div>
    """

    return html

@st.cache_data(ttl=CACHE_TTL)
def predict_reasoning(question: str, reasoning: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Send reasoning to API for analysis and return predictions.

    Args:
        question: The analytical question
        reasoning: The chain-of-thought reasoning to analyze
        model: The model to use for analysis

    Returns:
        Dictionary containing analysis results or None if API call fails
    """
    try:
        # Split reasoning into steps
        steps = [step.strip() for step in reasoning.split('\n') if step.strip()]

        # For demo/development purposes, generate mock data if API is not available
        try:
            response = requests.post(
                API_URL,
                json={"question": question, "reasoning": reasoning, "model": model},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            # Generate mock data for demonstration
            return generate_mock_predictions(question, steps, model)
    except Exception as e:
        logger.error(f"Error processing reasoning: {str(e)}")
        add_notification(f"Error processing reasoning: {str(e)}", "error")
        return None

def generate_mock_predictions(question: str, steps: List[str], model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Generate mock prediction data for demonstration purposes."""
    predictions = []

    # Adjust hallucination rate based on model
    if model == "standard":
        hallucination_rate = 0.2
        confidence_range = (0.65, 0.9)
    elif model == "advanced":
        hallucination_rate = 0.15
        confidence_range = (0.7, 0.95)
    else:  # expert
        hallucination_rate = 0.1
        confidence_range = (0.75, 0.98)

    hallucination_count = max(1, int(len(steps) * hallucination_rate)) if steps else 0

    # Randomly select steps to mark as hallucinations
    hallucination_indices = np.random.choice(
        range(len(steps)),
        size=hallucination_count,
        replace=False
    ).tolist() if steps else []

    for i, step in enumerate(steps):
        is_hallucination = i in hallucination_indices
        confidence = np.random.uniform(confidence_range[0], confidence_range[1])
        is_reliable = confidence > CONFIDENCE_THRESHOLDS["high"]

        # Generate explanation
        if is_hallucination:
            if is_reliable:
                explanation = "This statement contains factual errors or unsupported claims that contradict established knowledge."
            else:
                explanation = "This statement may contain speculative claims that aren't fully supported by evidence."
        else:
            if is_reliable:
                explanation = "This statement is well-supported by evidence and aligns with established knowledge."
            else:
                explanation = "This statement appears generally accurate but may benefit from additional verification."

        predictions.append({
            "step": step,
            "is_hallucination": is_hallucination,
            "confidence": confidence,
            "is_reliable": is_reliable,
            "explanation": explanation,
            "sources": generate_mock_sources(is_hallucination)
        })

    # Generate overall metrics
    avg_confidence = sum(pred["confidence"] for pred in predictions) / len(predictions) if predictions else 0
    reliability_score = sum(1 for pred in predictions if not pred["is_hallucination"]) / len(predictions) if predictions else 0

    return {
        "question": question,
        "num_steps": len(steps),
        "num_hallucinations": hallucination_count,
        "predictions": predictions,
        "metrics": {
            "avg_confidence": avg_confidence,
            "reliability_score": reliability_score * 100,
            "hallucination_rate": (hallucination_count / len(steps) * 100) if steps else 0,
            "model": model,
            "analysis_time": np.random.uniform(0.5, 2.5)
        }
    }

def generate_mock_sources(is_hallucination: bool) -> List[Dict[str, str]]:
    """Generate mock sources for predictions."""
    if is_hallucination:
        # Fewer sources for hallucinations
        num_sources = np.random.randint(0, 2)
    else:
        # More sources for grounded statements
        num_sources = np.random.randint(1, 4)

    sources = []
    potential_sources = [
        {"title": "IPCC Sixth Assessment Report", "url": "https://www.ipcc.ch/report/ar6/wg1/"},
        {"title": "NASA Climate Change Portal", "url": "https://climate.nasa.gov/"},
        {"title": "USDA Agricultural Projections", "url": "https://www.usda.gov/oce/commodity-markets/projections"},
        {"title": "National Climate Assessment", "url": "https://nca2018.globalchange.gov/"},
        {"title": "World Bank Climate Change Knowledge Portal", "url": "https://climateknowledgeportal.worldbank.org/"},
        {"title": "Project Management Institute Standards", "url": "https://www.pmi.org/pmbok-guide-standards"},
        {"title": "Harvard Business Review: Resource Allocation", "url": "https://hbr.org/topic/resource-allocation"},
        {"title": "Journal of Applied Cryptography", "url": "https://www.example.com/cryptography"},
        {"title": "WebAssembly Security Documentation", "url": "https://webassembly.org/docs/security/"}
    ]

    selected_indices = np.random.choice(len(potential_sources), size=num_sources, replace=False)
    for idx in selected_indices:
        sources.append(potential_sources[idx])

    return sources

def create_confidence_chart(predictions: List[Dict[str, Any]]) -> go.Figure:
    """Create a horizontal bar chart showing confidence scores for each step."""
    types = ["Grounded" if not pred["is_hallucination"] else "Hallucination" for pred in predictions]

    color_map = {
        "Grounded": "#10B981",
        "Hallucination": "#EF4444"
    }

    df = pd.DataFrame({
        "Step": [f"Step {i+1}" for i in range(len(predictions))],
        "Confidence": [pred["confidence"] for pred in predictions],
        "Type": types,
        "Text": [pred["step"][:50] + "..." if len(pred["step"]) > 50 else pred["step"] for pred in predictions],
    })

    fig = px.bar(
        df,
        x="Confidence",
        y="Step",
        color="Type",
        color_discrete_map=color_map,
        hover_data=["Text"],
        orientation='h',
        labels={"Confidence": "Confidence Score (0-1)", "Step": "", "Type": "Classification"},
        height=400,
    )

    fig.update_layout(
        font_family="Inter, sans-serif",
        title_font_family="Plus Jakarta Sans, sans-serif",
        plot_bgcolor="rgba(30, 41, 59, 0.8)",
        paper_bgcolor="rgba(30, 41, 59, 0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            range=[0, 1],
            gridcolor="rgba(203, 213, 225, 0.1)",
            zerolinecolor="rgba(203, 213, 225, 0.1)",
            tickfont=dict(color="#CBD5E1"),
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(203, 213, 225, 0.1)",
            tickfont=dict(color="#CBD5E1"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1"),
            bgcolor="rgba(30, 41, 59, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.1)"
        ),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(203, 213, 225, 0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(203, 213, 225, 0.1)')

    return fig

def create_distribution_chart(predictions: List[Dict[str, Any]]) -> go.Figure:
    """Create a histogram showing the distribution of confidence scores."""
    confidences = [pred["confidence"] for pred in predictions]
    types = ["Grounded" if not pred["is_hallucination"] else "Hallucination" for pred in predictions]

    df = pd.DataFrame({
        "Confidence": confidences,
        "Type": types,
    })

    fig = go.Figure()

    for type_name, color in [("Grounded", "#10B981"), ("Hallucination", "#EF4444")]:
        type_data = df[df["Type"] == type_name]
        if not type_data.empty:
            fig.add_trace(go.Histogram(
                x=type_data["Confidence"],
                name=type_name,
                marker_color=color,
                opacity=0.8,
                nbinsx=10,
            ))

    fig.update_layout(
        title="Confidence Score Distribution",
        title_font=dict(size=18, family="Plus Jakarta Sans, sans-serif", color="#F8FAFC"),
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        barmode='overlay',
        plot_bgcolor="rgba(30, 41, 59, 0.8)",
        paper_bgcolor="rgba(30, 41, 59, 0)",
        font_family="Inter, sans-serif",
        font_color="#CBD5E1",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1"),
            bgcolor="rgba(30, 41, 59, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.1)"
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(203, 213, 225, 0.1)',
        zerolinecolor="rgba(203, 213, 225, 0.1)",
        tickfont=dict(color="#CBD5E1"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(203, 213, 225, 0.1)',
        zerolinecolor="rgba(203, 213, 225, 0.1)",
        tickfont=dict(color="#CBD5E1"),
    )

    return fig

def create_reasoning_flow_chart(predictions: List[Dict[str, Any]]) -> go.Figure:
    """Create a flow chart showing the reasoning steps and their relationships."""
    # Create a graph
    G = nx.DiGraph()

    # Add nodes for each step
    for i, pred in enumerate(predictions):
        node_id = f"Step {i+1}"
        is_hallucination = pred["is_hallucination"]
        confidence = pred["confidence"]

        if is_hallucination:
            if confidence > CONFIDENCE_THRESHOLDS["high"]:
                color = "#EF4444"  # danger
            else:
                color = "#F59E0B"  # warning
        else:
            if confidence > CONFIDENCE_THRESHOLDS["high"]:
                color = "#10B981"  # success
            else:
                color = "#6366F1"  # primary

        G.add_node(node_id, color=color, confidence=confidence, text=pred["step"])

        # Add edge from previous step if not the first step
        if i > 0:
            G.add_edge(f"Step {i}", node_id)

    # Create a PyVis network
    net = Network(height="500px", width="100%", bgcolor="#1E293B", font_color="#CBD5E1")

    # Add nodes and edges
    for node in G.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        net.add_node(
            node_id,
            label=node_id,
            title=node_data["text"][:100] + "..." if len(node_data["text"]) > 100 else node_data["text"],
            color=node_data["color"],
            size=25 + node_data["confidence"] * 10
        )

    for edge in G.edges():
        net.add_edge(edge[0], edge[1])

    # Set options
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 1000
            }
        },
        "edges": {
            "color": {
                "inherit": true
            },
            "smooth": {
                "enabled": true,
                "type": "dynamic"
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200
        }
    }
    """)

    # Generate HTML file
    html_path = "cache/reasoning_flow.html"
    net.save_graph(html_path)

    # Read the HTML file
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content

def generate_insights(predictions: List[Dict[str, Any]], metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate AI insights based on the analysis results."""
    insights = []

    # Count hallucinations and calculate average confidence
    hallucinations = [p for p in predictions if p["is_hallucination"]]
    grounded = [p for p in predictions if not p["is_hallucination"]]

    # Insight 1: Overall assessment
    if len(hallucinations) == 0:
        insights.append({
            "title": "Strong Reasoning Chain",
            "content": "The reasoning chain appears robust with consistent grounding across all steps. No hallucinations were detected.",
            "type": "success"
        })
    elif len(hallucinations) / len(predictions) > 0.3:
        insights.append({
            "title": "Significant Issues Detected",
            "content": f"A substantial portion ({len(hallucinations) / len(predictions):.0%}) of the reasoning chain contains potential issues. Consider revising the entire analysis.",
            "type": "danger"
        })
    else:
        insights.append({
            "title": "Generally Sound Reasoning",
            "content": f"The reasoning chain is mostly well-grounded with {len(hallucinations)} potential issues identified out of {len(predictions)} steps.",
            "type": "info"
        })

    # Insight 2: High confidence hallucinations
    high_conf_hallucinations = [p for p in hallucinations if p["confidence"] > CONFIDENCE_THRESHOLDS["high"]]
    if high_conf_hallucinations:
        insights.append({
            "title": "Confident Errors Detected",
            "content": f"Found {len(high_conf_hallucinations)} high-confidence errors, suggesting the model may be confidently wrong in some assertions.",
            "type": "danger"
        })

    # Insight 3: Low confidence grounded statements
    low_conf_grounded = [p for p in grounded if p["confidence"] < CONFIDENCE_THRESHOLDS["medium"]]
    if low_conf_grounded:
        insights.append({
            "title": "Uncertain Grounded Claims",
            "content": f"{len(low_conf_grounded)} grounded statements have lower confidence scores. Consider validating these with additional sources.",
            "type": "warning"
        })

    # Insight 4: Model-specific insight
    model = metrics.get("model", DEFAULT_MODEL)
    if model == "standard":
        insights.append({
            "title": "Consider Advanced Analysis",
            "content": "For more nuanced verification, consider using the Advanced or Expert model which can detect subtler reasoning errors.",
            "type": "info"
        })
    elif model == "expert":
        insights.append({
            "title": "Expert-Level Analysis",
            "content": "You're using our most sophisticated verification model, providing the highest confidence in the analysis results.",
            "type": "success"
        })

    # Insight 5: Structural insight
    if len(predictions) > 7:
        insights.append({
            "title": "Complex Reasoning Chain",
            "content": "This is a complex reasoning chain with many steps. Consider breaking it into smaller, more focused analyses for better verification.",
            "type": "info"
        })
    elif len(predictions) < 3:
        insights.append({
            "title": "Limited Reasoning Steps",
            "content": "This reasoning chain has few steps. More detailed reasoning with intermediate steps may improve verifiability.",
            "type": "warning"
        })

    # Return top 5 insights
    return insights[:5]

def save_to_history(question: str, reasoning: str, results: Dict[str, Any]) -> None:
    """Save analysis results to history for future reference."""
    if "history" not in st.session_state:
        st.session_state.history = []

    # Create a timestamp
    timestamp = datetime.now()

    # Create a unique ID for this analysis
    analysis_id = str(uuid.uuid4())

    # Add to history
    st.session_state.history.insert(0, {
        "id": analysis_id,
        "timestamp": timestamp,
        "question": question,
        "reasoning": reasoning,
        "results": results,
        "model": results.get("metrics", {}).get("model", DEFAULT_MODEL),
        "favorite": False
    })

    # Keep only the max number of items
    max_history = st.session_state.settings.get("max_history", MAX_HISTORY_ITEMS)
    if len(st.session_state.history) > max_history:
        st.session_state.history = st.session_state.history[:max_history]

    # Add notification
    add_notification("Analysis saved to history", "success")

def toggle_favorite(analysis_id: str) -> None:
    """Toggle favorite status for an analysis."""
    for item in st.session_state.history:
        if item["id"] == analysis_id:
            item["favorite"] = not item["favorite"]

            if item["favorite"]:
                st.session_state.favorites.append(item["id"])
                add_notification(f"Analysis added to favorites", "success")
            else:
                if item["id"] in st.session_state.favorites:
                    st.session_state.favorites.remove(item["id"])
                add_notification(f"Analysis removed from favorites", "info")

            break

def export_results(results: Dict[str, Any], format_type: str) -> Tuple[bytes, str, str]:
    """Export analysis results in various formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reasoning_analysis_{timestamp}"

    if format_type == "json":
        content = json.dumps(results, indent=2, default=str).encode('utf-8')
        mime_type = "application/json"
        filename += ".json"
    elif format_type == "csv":
        # Create a CSV string
        csv_data = "Step,Classification,Confidence,Explanation\n"
        for i, pred in enumerate(results["predictions"]):
            classification = "Hallucination" if pred["is_hallucination"] else "Grounded"
            explanation = pred["explanation"].replace(",", ";")  # Avoid CSV issues
            csv_data += f"{i+1},{classification},{pred['confidence']:.2f},\"{explanation}\"\n"
        content = csv_data.encode('utf-8')
        mime_type = "text/csv"
        filename += ".csv"
    elif format_type == "pdf":
        # Create a PDF report
        pdf = FPDF()
        pdf.add_page()

        # Set font
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Reasoning Analysis Report", ln=True, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Question: {results['question']}", ln=True)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)

        # Summary
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Summary", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Total Steps: {results['num_steps']}", ln=True)
        pdf.cell(0, 10, f"Hallucinations: {results['num_hallucinations']}", ln=True)
        pdf.cell(0, 10, f"Error Rate: {(results['num_hallucinations'] / results['num_steps'] * 100):.1f}%", ln=True)
        pdf.ln(5)

        # Step-by-Step Analysis
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Step-by-Step Analysis", ln=True)

        for i, pred in enumerate(results["predictions"]):
            classification = "Hallucination" if pred["is_hallucination"] else "Grounded"
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Step {i+1}: {classification} (Confidence: {pred['confidence']:.2f})", ln=True)

            pdf.set_font("Arial", "", 12)
            # Split long text into multiple lines
            step_text = pred["step"]
            pdf.multi_cell(0, 10, step_text)

            pdf.set_font("Arial", "I", 12)
            pdf.multi_cell(0, 10, f"Explanation: {pred['explanation']}")
            pdf.ln(5)

        # Save to BytesIO
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        content = pdf_output.getvalue()
        mime_type = "application/pdf"
        filename += ".pdf"
    else:  # text
        # Create a text report
        text = f"Analysis Report for: {results['question']}\n"
        text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        text += f"Total Steps: {results['num_steps']}\n"
        text += f"Hallucinations: {results['num_hallucinations']}\n"
        text += f"Error Rate: {(results['num_hallucinations'] / results['num_steps'] * 100):.1f}%\n\n"

        text += "Step-by-Step Analysis:\n"
        for i, pred in enumerate(results["predictions"]):
            classification = "Hallucination" if pred["is_hallucination"] else "Grounded"
            text += f"Step {i+1}: {classification} (Confidence: {pred['confidence']:.2f})\n"
            text += f"  {pred['step']}\n"
            text += f"  Explanation: {pred['explanation']}\n\n"

        content = text.encode('utf-8')
        mime_type = "text/plain"
        filename += ".txt"

    return content, mime_type, filename

def add_notification(message: str, type: str = "info", duration: int = 5):
    """Add a notification to the session state."""
    if not st.session_state.settings.get("notifications_enabled", True):
        return

    notification_id = str(uuid.uuid4())
    st.session_state.notifications.append({
        "id": notification_id,
        "message": message,
        "type": type,
        "timestamp": datetime.now(),
        "duration": duration
    })

    # Limit the number of notifications
    if len(st.session_state.notifications) > 5:
        st.session_state.notifications.pop(0)

def show_notifications():
    """Display notifications in the UI."""
    if not st.session_state.notifications:
        return

    current_time = datetime.now()
    notifications_to_keep = []

    for notification in st.session_state.notifications:
        time_diff = (current_time - notification["timestamp"]).total_seconds()
        if time_diff < notification["duration"]:
            # Still valid notification
            with st.container():
                if notification["type"] == "success":
                    st.success(notification["message"], icon="✅")
                elif notification["type"] == "error":
                    st.error(notification["message"], icon="❌")
                elif notification["type"] == "warning":
                    st.warning(notification["message"], icon="⚠️")
                else:
                    st.info(notification["message"], icon="ℹ️")
            notifications_to_keep.append(notification)

    # Update notifications list
    st.session_state.notifications = notifications_to_keep

def render_header():
    """Render the application header."""
    st.markdown(
        f"""
        <div class="app-container">
            <div class="app-header">
                <div class="header-left">
                    <div class="logo-container">
                        <img src="{get_logo_base64()}" class="logo-icon" alt="Logo">
                        <h1 class="app-title">Reasoning Verifier <span class="-badge"></span></h1>
                    </div>
                    <p class="header-subtitle">Advanced validation engine for chain-of-thought reasoning in AI systems.</p>
                </div>
                <div class="theme-toggle">
                    <button class="{'active' if st.session_state.theme == 'dark' else ''}" onclick="changeTheme('dark')">Dark</button>
                    <button class="{'active' if st.session_state.theme == 'light' else ''}" onclick="changeTheme('light')">Light</button>
                    <button class="{'active' if st.session_state.theme == 'high_contrast' else ''}" onclick="changeTheme('high_contrast')">High Contrast</button>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )


def render_dashboard():
    """Render the dashboard page."""
    colored_header(
        label="Dashboard",
        description="Overview of your reasoning verification activities",
        color_name="violet-70"
    )

    # Key metrics
    st.subheader("Key Metrics")

    metrics_cols = st.columns(4)

    with metrics_cols[0]:
        st.metric(
            label="Total Analyses",
            value=len(st.session_state.history),
            delta="+3 today",
            help="Total number of reasoning analyses performed"
        )

    with metrics_cols[1]:
        # Calculate average hallucination rate across all analyses
        if st.session_state.history:
            avg_hallucination_rate = np.mean([
                h["results"].get("metrics", {}).get("hallucination_rate", 0)
                for h in st.session_state.history if "results" in h
            ])
        else:
            avg_hallucination_rate = 0

        st.metric(
            label="Avg. Hallucination Rate",
            value=f"{avg_hallucination_rate:.1f}%",
            delta="-2.5%" if avg_hallucination_rate < 20 else "+2.5%",
            delta_color="inverse",
            help="Average hallucination rate across all analyses"
        )

    with metrics_cols[2]:
        # Calculate average confidence score
        if st.session_state.history:
            avg_confidence = np.mean([
                h["results"].get("metrics", {}).get("avg_confidence", 0)
                for h in st.session_state.history if "results" in h
            ])
        else:
            avg_confidence = 0

        st.metric(
            label="Avg. Confidence",
            value=f"{avg_confidence:.2f}",
            delta="+0.05" if avg_confidence > 0.8 else "-0.05",
            help="Average confidence score across all analyses"
        )

    with metrics_cols[3]:
        # Count favorite analyses
        favorites_count = len(st.session_state.favorites)

        st.metric(
            label="Saved Favorites",
            value=favorites_count,
            delta="+1" if favorites_count > 0 else None,
            help="Number of analyses saved as favorites"
        )

    # Recent activity
    st.subheader("Recent Activity")

    if not st.session_state.history:
        st.info("No recent activity. Start by analyzing some reasoning chains!")
    else:
        # Show recent analyses in a table
        recent_data = []
        for item in st.session_state.history[:5]:  # Show only 5 most recent
            recent_data.append({
                "Date": item["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Question": item["question"][:50] + "..." if len(item["question"]) > 50 else item["question"],
                "Model": item["model"],
                "Hallucinations": item["results"].get("num_hallucinations", 0),
                "Steps": item["results"].get("num_steps", 0)
            })

        recent_df = pd.DataFrame(recent_data)
        st.dataframe(recent_df, use_container_width=True)

    # Charts
    st.subheader("Analytics")

    chart_cols = st.columns(2)

    with chart_cols[0]:
        with chart_container(chart_type="altair"):
            if st.session_state.history:
                # Prepare data for hallucination rate over time
                chart_data = []
                for item in st.session_state.history:
                    if "results" in item and "metrics" in item["results"]:
                        chart_data.append({
                            "date": item["timestamp"],
                            "hallucination_rate": item["results"]["metrics"].get("hallucination_rate", 0)
                        })

                if chart_data:
                    chart_df = pd.DataFrame(chart_data)
                    chart = alt.Chart(chart_df).mark_line(point=True).encode(
                        x=alt.X('date:T', title='Date'),
                        y=alt.Y('hallucination_rate:Q', title='Hallucination Rate (%)', scale=alt.Scale(domain=[0, 100])),
                        tooltip=['date:T', 'hallucination_rate:Q']
                    ).properties(
                        title='Hallucination Rate Over Time'
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Not enough data to generate chart")
            else:
                st.info("No data available for visualization")

    with chart_cols[1]:
        with chart_container(chart_type="altair"):
            if st.session_state.history:
                # Prepare data for model usage
                model_counts = {}
                for item in st.session_state.history:
                    model = item.get("model", "Unknown")
                    model_counts[model] = model_counts.get(model, 0) + 1

                if model_counts:
                    model_df = pd.DataFrame([
                        {"model": model, "count": count}
                        for model, count in model_counts.items()
                    ])

                    chart = alt.Chart(model_df).mark_bar().encode(
                        x=alt.X('model:N', title='Model'),
                        y=alt.Y('count:Q', title='Number of Analyses'),
                        color=alt.Color('model:N', legend=None),
                        tooltip=['model:N', 'count:Q']
                    ).properties(
                        title='Model Usage'
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Not enough data to generate chart")
            else:
                st.info("No data available for visualization")

    # Quick actions
    st.subheader("Quick Actions")

    action_cols = st.columns(3)

    with action_cols[0]:
        if st.button("New Analysis", use_container_width=True):
            st.session_state.current_page = "analyzer"
            st.rerun()

    with action_cols[1]:
        if st.button("View History", use_container_width=True):
            st.session_state.current_page = "history"
            st.rerun()

    with action_cols[2]:
        if st.button("Batch Analysis", use_container_width=True):
            st.session_state.current_page = "batch"
            st.rerun()

def render_analyzer():
    """Render the analyzer page."""
    colored_header(
        label="Reasoning Analyzer",
        description="Analyze chain-of-thought reasoning for potential hallucinations",
        color_name="violet-70"
    )

    # Input section
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M11 4H4V11H11V4Z" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M20 4H13V11H20V4Z" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M11 13H4V20H11V13Z" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M20 13H13V20H20V13Z" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            Input Analysis
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="content-card">', unsafe_allow_html=True)

        # Create tabs for different input methods
        input_tab, upload_tab, example_tab = st.tabs(["Manual Input", "Upload File", "Examples"])

        with input_tab:
            # Input fields
            question = st.text_input("Question", value=st.session_state.question, placeholder="Enter the analytical question here...")
            reasoning = st.text_area("Chain-of-Thought Reasoning", value=st.session_state.reasoning, placeholder="Enter reasoning steps here (one step per line)...", height=200)

            # Model selection
            col1, col2 = st.columns([3, 1])
            with col1:
                model = st.selectbox(
                    "Verification Model",
                    options=AVAILABLE_MODELS,
                    index=AVAILABLE_MODELS.index(st.session_state.model) if st.session_state.model in AVAILABLE_MODELS else 0,
                    help="Select the model to use for verification. Advanced and Expert models provide more nuanced analysis but may take longer."
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                analyze_button = st.button("ANALYZE REASONING", use_container_width=True)

        with upload_tab:
            uploaded_file = st.file_uploader("Upload a text file containing your reasoning", type=["txt", "md", "json"])

            if uploaded_file is not None:
                try:
                    # Try to parse as JSON first
                    try:
                        content = json.load(uploaded_file)
                        if isinstance(content, dict):
                            if "question" in content and "reasoning" in content:
                                question = content["question"]
                                reasoning = content["reasoning"]
                                st.success("Successfully loaded question and reasoning from JSON file.")
                            else:
                                st.warning("JSON file doesn't contain 'question' and 'reasoning' fields.")
                        else:
                            st.warning("JSON file doesn't contain a valid object.")
                    except:
                        # If not JSON, treat as plain text
                        content = uploaded_file.getvalue().decode("utf-8")
                        lines = content.strip().split("\n")

                        if len(lines) >= 2:
                            question = lines[0]
                            reasoning = "\n".join(lines[1:])
                            st.success("Successfully loaded question and reasoning from text file.")
                        else:
                            st.warning("Text file should contain at least 2 lines (question and reasoning).")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

            # Model selection
            col1, col2 = st.columns([3, 1])
            with col1:
                model = st.selectbox(
                    "Verification Model",
                    options=AVAILABLE_MODELS,
                    index=AVAILABLE_MODELS.index(st.session_state.model) if st.session_state.model in AVAILABLE_MODELS else 0,
                    key="upload_model",
                    help="Select the model to use for verification. Advanced and Expert models provide more nuanced analysis but may take longer."
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                analyze_upload_button = st.button("ANALYZE UPLOADED", use_container_width=True)

                if analyze_upload_button:
                    analyze_button = True
                else:
                    analyze_button = False

        with example_tab:
            examples = load_example_data()

            example_type = st.radio(
                "Select an example",
                options=["simple", "complex", "technical"],
                format_func=lambda x: {
                    "simple": "Simple Resource Allocation",
                    "complex": "Climate Change Impact Analysis",
                    "technical": "WebAssembly Security Analysis"
                }.get(x, x)
            )

            st.markdown("### Example Question")
            st.info(examples[example_type]["question"])

            st.markdown("### Example Reasoning")
            st.info(examples[example_type]["reasoning"])

            # Model selection
            col1, col2 = st.columns([3, 1])
            with col1:
                model = st.selectbox(
                    "Verification Model",
                    options=AVAILABLE_MODELS,
                    index=AVAILABLE_MODELS.index(st.session_state.model) if st.session_state.model in AVAILABLE_MODELS else 0,
                    key="example_model",
                    help="Select the model to use for verification. Advanced and Expert models provide more nuanced analysis but may take longer."
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                load_example_button = st.button("USE THIS EXAMPLE", use_container_width=True)

                if load_example_button:
                    question = examples[example_type]["question"]
                    reasoning = examples[example_type]["reasoning"]
                    analyze_button = True
                else:
                    analyze_button = False

        st.markdown('</div>', unsafe_allow_html=True)

    # Process analysis
    if analyze_button:
        if not question or not reasoning:
            st.error("Please provide both a question and reasoning steps for analysis.")
        else:
            with st.spinner("Processing reasoning chain..."):
                # Display loading animation
                lottie_loading = get_lottie_animation("loading")
                st_lottie(lottie_loading, height=150, key="loading")

                # Get predictions
                response = predict_reasoning(question, reasoning, model)

                if response:
                    # Save to history if auto-save is enabled
                    if st.session_state.settings.get("auto_save", True):
                        save_to_history(question, reasoning, response)

                    # Store in session state
                    st.session_state.results = response
                    st.session_state.show_results = True
                    st.session_state.question = question
                    st.session_state.reasoning = reasoning
                    st.session_state.model = model

                    # Show success animation
                    lottie_success = get_lottie_animation("success")
                    st_lottie(lottie_success, height=150, key="success")

                    # Add notification
                    add_notification("Analysis completed successfully", "success")

                    # Rerun to clear the spinner and show results
                    st.rerun()

    # Display results if available
    if st.session_state.show_results and st.session_state.results:
        render_results(st.session_state.results, st.session_state.question)

def render_results(response: Dict[str, Any], question: str):
    """Render the analysis results."""
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M7 10L12 15L17 10" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M12 15V3" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            Analysis Results
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f'<div class="content-card" style="margin-bottom: 1.5rem;"><strong style="color: var(--text-primary);">Original Question:</strong> <span style="color: var(--text-secondary);">{question}</span></div>', unsafe_allow_html=True)

    # Get metrics
    num_steps = response["num_steps"]
    num_hallucinations = response["num_hallucinations"]
    hallucination_rate = (num_hallucinations / num_steps) * 100 if num_steps > 0 else 0
    metrics = response.get("metrics", {})
    avg_confidence = metrics.get("avg_confidence", sum(pred["confidence"] for pred in response["predictions"]) / num_steps if num_steps > 0 else 0)
    reliability_score = metrics.get("reliability_score", 100 - hallucination_rate)
    analysis_time = metrics.get("analysis_time", 1.0)

    # Metrics cards with improved styling
    style_metric_cards()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="TOTAL STEPS",
            value=num_steps,
            delta=None,
            help="Total number of reasoning steps analyzed"
        )

    with col2:
        st.metric(
            label="HALLUCINATIONS",
            value=num_hallucinations,
            delta=None,
            delta_color="inverse",
            help="Number of potential hallucinations detected"
        )

    with col3:
        st.metric(
            label="ERROR RATE",
            value=f"{hallucination_rate:.1f}%",
            delta=None,
            delta_color="inverse",
            help="Percentage of steps identified as hallucinations"
        )

    with col4:
        st.metric(
            label="RELIABILITY SCORE",
            value=f"{reliability_score:.1f}%",
            delta=None,
            help="Overall reliability score of the reasoning chain"
        )

    # Tabs for different views of the results
    tab1, tab2, tab3, tab4 = st.tabs(["Step-by-Step Analysis", "Visualizations", "AI Insights", "Technical Details"])

    with tab1:
        st.markdown(
            """
            <div class="section-header">
                <div class="section-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M9 18L15 12L9 6" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                Step-by-Step Verification
            </div>
            """,
            unsafe_allow_html=True
        )

        for i, pred in enumerate(response["predictions"]):
            step_html = format_step(
                pred["step"],
                pred["is_hallucination"],
                pred["confidence"],
                pred["is_reliable"],
            )
            st.markdown(step_html, unsafe_allow_html=True)

            # Expandable details
            with st.expander("View explanation and sources"):
                st.markdown(f"**Explanation:** {pred['explanation']}")

                if pred.get("sources"):
                    st.markdown("**Sources:**")
                    for source in pred["sources"]:
                        st.markdown(f"- [{source['title']}]({source['url']})")
                else:
                    st.info("No specific sources available for this step.")

    with tab2:
        st.markdown(
            """
            <div class="section-header">
                <div class="section-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M7 12L10 15L17 8M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                Confidence Analysis
            </div>
            """,
            unsafe_allow_html=True
        )

        # Visualization tabs
        viz_tab1, viz_tab2 = st.tabs(["Confidence Scores", "Distribution"])

        with viz_tab1:
            with chart_container(chart_type="plotly"):
                confidence_fig = create_confidence_chart(response["predictions"])
                st.plotly_chart(confidence_fig, use_container_width=True)

        with viz_tab2:
            with chart_container(chart_type="plotly"):
                distribution_fig = create_distribution_chart(response["predictions"])
                st.plotly_chart(distribution_fig, use_container_width=True)

    with tab3:
        # Generate AI insights
        insights = generate_insights(response["predictions"], metrics)

        st.markdown(
            f"""
            <div class="insights-container">
                <div class="insights-header">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                        <path d="M9.66347 17H14.3364M11.9999 3V4M18.3639 5.63604L17.6568 6.34315M21 11.9999H20M4 11.9999H3M6.34309 6.34315L5.63599 5.63604M8.46441 15.5356C6.51179 13.5829 6.51179 10.4171 8.46441 8.46449C10.417 6.51187 13.5829 6.51187 15.5355 8.46449C17.4881 10.4171 17.4881 13.5829 15.5355 15.5356L14.9884 16.0827C14.3555 16.7155 13.9999 17.5739 13.9999 18.469V19C13.9999 20.1046 13.1045 21 11.9999 21C10.8954 21 9.99995 20.1046 9.99995 19V18.469C9.99995 17.5739 9.6444 16.7155 9.01151 16.0827L8.46441 15.5356Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    AI Analysis Insights
                </div>
            """,
            unsafe_allow_html=True
        )

        # Display insights in cards
        insight_grid = grid(3, vertical_align="start")

        for insight in insights:
            with insight_grid.container():
                st.markdown(f"### {insight['title']}")
                st.markdown(f"{insight['content']}")

                # Add colored indicator based on insight type
                if insight["type"] == "success":
                    st.markdown('<div style="height: 4px; background: linear-gradient(90deg, #10B981, #34D399); margin-top: 10px;"></div>', unsafe_allow_html=True)
                elif insight["type"] == "danger":
                    st.markdown('<div style="height: 4px; background: linear-gradient(90deg, #EF4444, #F87171); margin-top: 10px;"></div>', unsafe_allow_html=True)
                elif insight["type"] == "warning":
                    st.markdown('<div style="height: 4px; background: linear-gradient(90deg, #F59E0B, #FBBF24); margin-top: 10px;"></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="height: 4px; background: linear-gradient(90deg, #3B82F6, #60A5FA); margin-top: 10px;"></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown("### Technical Details")

        # Model information
        st.markdown("#### Model Information")
        st.json({
            "model": metrics.get("model", "Standard"),
            "version": "1.0.0",
            "analysis_time": f"{analysis_time:.2f} seconds",
            "confidence_threshold": CONFIDENCE_THRESHOLDS["high"]
        })

        # Raw predictions
        st.markdown("#### Raw Predictions")
        with st.expander("View raw prediction data"):
            st.json(response)

        # Export options
        st.markdown("#### Export Options")
        export_col1, export_col2 = st.columns([3, 1])

        with export_col1:
            export_format = st.selectbox(
                "Export Format",
                options=["json", "csv", "text", "pdf"],
                format_func=lambda x: {
                    "json": "JSON",
                    "csv": "CSV",
                    "text": "Text Report",
                    "pdf": "PDF Report"
                }.get(x, x)
            )

        with export_col2:
            if st.button("Export", use_container_width=True):
                content, mime_type, filename = export_results(response, export_format)
                st.download_button(
                    label="Download",
                    data=content,
                    file_name=filename,
                    mime=mime_type,
                    key="download_button"
                )

def render_history():
    """Render the history page."""
    colored_header(
        label="Analysis History",
        description="View and manage your past reasoning analyses",
        color_name="violet-70"
    )

    if not st.session_state.history:
        st.info("No analysis history yet. Run your first analysis to see it here.")

        if st.button("Go to Analyzer", use_container_width=False):
            st.session_state.current_page = "analyzer"
            st.rerun()

        return

    # Filters
    with st.expander("Filters", expanded=False):
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            search_term = st.text_input(
                "Search",
                value=st.session_state.filters["search_term"],
                placeholder="Search by question or content..."
            )

            model_filter = st.multiselect(
                "Model",
                options=AVAILABLE_MODELS,
                default=[]
            )

        with filter_col2:
            date_range = st.date_input(
                "Date Range",
                value=st.session_state.filters["date_range"],
                max_value=datetime.now()
            )

            hallucination_range = st.slider(
                "Hallucination Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=(0.0, 100.0),
                step=5.0
            )

        # Update filters in session state
        st.session_state.filters["search_term"] = search_term
        st.session_state.filters["date_range"] = date_range
        st.session_state.filters["model_filter"] = model_filter
        st.session_state.filters["hallucination_range"] = hallucination_range

    # Apply filters
    filtered_history = st.session_state.history

    if search_term:
        filtered_history = [
            h for h in filtered_history
            if search_term.lower() in h["question"].lower() or
               search_term.lower() in h["reasoning"].lower()
        ]

    if model_filter:
        filtered_history = [
            h for h in filtered_history
            if h.get("model", "").lower() in [m.lower() for m in model_filter]
        ]

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())

        filtered_history = [
            h for h in filtered_history
            if start_datetime <= h["timestamp"] <= end_datetime
        ]

    if hallucination_range and len(hallucination_range) == 2:
        min_rate, max_rate = hallucination_range

        filtered_history = [
            h for h in filtered_history
            if min_rate <= h["results"].get("metrics", {}).get("hallucination_rate", 0) <= max_rate
        ]

    # Display history
    if not filtered_history:
        st.warning("No items match your filters. Try adjusting your criteria.")
    else:
        # Create a table of history items
        history_data = []
        for i, item in enumerate(filtered_history):
            hallucination_rate = item["results"].get("metrics", {}).get("hallucination_rate", 0)
            history_data.append({
                "Date": item["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Question": item["question"][:50] + "..." if len(item["question"]) > 50 else item["question"],
                "Model": item["model"],
                "Hallucination Rate": f"{hallucination_rate:.1f}%",
                "Steps": item["results"].get("num_steps", 0),
                "Favorite": "★" if item.get("favorite", False) else "",
                "id": item["id"]  # Hidden column for reference
            })

        history_df = pd.DataFrame(history_data)

        # Display the table with selection
        selected_indices = st.dataframe(
            history_df.drop(columns=["id"]),
            use_container_width=True,
            column_config={
                "Favorite": st.column_config.Column(
                    "Favorite",
                    width="small",
                    help="Starred items"
                )
            },
            height=400
        )

        # Action buttons
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)

        with action_col1:
            if st.button("View Details", use_container_width=True):
                if selected_indices is not None:
                    selected_item = filtered_history[selected_indices]
                    st.session_state.question = selected_item["question"]
                    st.session_state.reasoning = selected_item["reasoning"]
                    st.session_state.results = selected_item["results"]
                    st.session_state.show_results = True
                    st.session_state.model = selected_item.get("model", DEFAULT_MODEL)
                    st.session_state.current_page = "analyzer"
                    st.rerun()

        with action_col2:
            if st.button("Toggle Favorite", use_container_width=True):
                if selected_indices is not None:
                    selected_item = filtered_history[selected_indices]
                    toggle_favorite(selected_item["id"])
                    st.rerun()

        with action_col3:
            if st.button("Compare", use_container_width=True):
                if selected_indices is not None:
                    selected_item = filtered_history[selected_indices]
                    if selected_item["id"] not in [item["id"] for item in st.session_state.comparison_items]:
                        st.session_state.comparison_items.append(selected_item)
                        add_notification(f"Added to comparison", "success")
                    else:
                        add_notification(f"Item already in comparison", "warning")

                    if len(st.session_state.comparison_items) >= 2:
                        st.session_state.current_page = "compare"
                        st.rerun()

        with action_col4:
            if st.button("Delete", use_container_width=True):
                if selected_indices is not None:
                    selected_item = filtered_history[selected_indices]
                    st.session_state.history = [item for item in st.session_state.history if item["id"] != selected_item["id"]]
                    add_notification(f"Analysis deleted", "info")
                    st.rerun()
def render_footer() -> None:
    """Render the application footer."""
    st.markdown(
        f"""
        <div class="custom-footer">
            <p>Reasoning Verifier © {datetime.now().year} | Version {VERSION}</p>
            <div class="footer-links">
                <a href="#" class="footer-link">Documentation</a>
                <a href="#" class="footer-link">Terms of Service</a>
                <a href="#" class="footer-link">Privacy Policy</a>
                <a href="#" class="footer-link">Support</a>
            </div>
        </div>
        </div>


          """,
        unsafe_allow_html=True
    )

def main():
    """Main application function."""
    # Apply custom CSS
    local_css()

    # Apply theme
    apply_theme()

    # Check authentication
    if not authenticate_user():
        return

    # Show notifications
    show_notifications()

    # Render header
    render_header()

    # Render sidebar navigation
    with st.sidebar:
        # User profile
        if st.session_state.authenticated:
            st.markdown(
                f"""
                <div class="user-profile">
                    <div class="user-avatar">{st.session_state.user_name[0].upper()}</div>
                    <div class="user-info">
                        <div class="user-name">{st.session_state.user_name}</div>
                        <div class="user-role">{st.session_state.user_role}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Navigation
        selected = option_menu(
            menu_title="Navigation",
            options=["Dashboard", "Analyzer", "History", "Batch Analysis", "Compare", "Settings"],
            icons=["house", "search", "clock-history", "list-task", "arrow-left-right", "gear"],
            menu_icon="cast",
            default_index=["dashboard", "analyzer", "history", "batch", "compare", "settings"].index(st.session_state.current_page),
            orientation="vertical",
        )

        if selected.lower() != st.session_state.current_page:
            st.session_state.current_page = selected.lower()
            st.rerun()

    # Render the appropriate page based on current_page
    if st.session_state.current_page == "dashboard":
        render_dashboard()
    elif st.session_state.current_page == "analyzer":
        render_analyzer()
    elif st.session_state.current_page == "history":
        render_history()
    elif st.session_state.current_page == "batch":
        render_batch_analysis()
    elif st.session_state.current_page == "compare":
        render_comparison()
    elif st.session_state.current_page == "settings":
        render_settings()

    # Render footer
    render_footer()

if __name__ == "__main__":
    main()
