"""
========================================================
 MediAlt — Medical Prescription Analyzer & Brand Suggester
 File: app.py
 Run: streamlit run app.py

 PURPOSE:
   This is the main entry point of the web application.
   It builds the entire user interface using Streamlit,
   connects the two AI models (Donut OCR + Gemini), and
   displays the results in a professional dashboard.

 HOW IT FITS IN THE PROJECT:
   app.py  ──calls──►  src/ocr_engine.py      (Donut OCR)
   app.py  ──calls──►  src/gemini_analyzer.py (Gemini AI)
   Both of those read from the local model/ folder.
========================================================
"""

# ─────────────────────────────────────────────────────
# SECTION 1: STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────────────

import os       # os lets us read environment variables and file paths
                # e.g., os.getenv("GEMINI_API_KEY") reads from the .env file

import sys      # sys lets us modify Python's module search path
                # We use it to make sure Python can find our src/ folder

import streamlit as st   # Streamlit is the web UI framework
                         # It turns Python scripts into interactive web apps
                         # with no HTML/JS knowledge required

from PIL import Image    # PIL (Pillow) is a Python image processing library
                         # Image.open() reads a JPG/PNG file into memory
                         # .convert("RGB") makes sure colors are in standard format

from dotenv import load_dotenv   # python-dotenv reads key=value pairs from .env file
                                 # This lets us store secrets (like API keys) safely
                                 # outside the code


# ─────────────────────────────────────────────────────
# SECTION 2: INITIAL SETUP (runs once when app starts)
# ─────────────────────────────────────────────────────

load_dotenv()
# WHY: Reads the .env file and loads GEMINI_API_KEY into environment variables.
# After this line, os.getenv("GEMINI_API_KEY") will return your actual API key.
# Without this, the key field in the sidebar would be empty on startup.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# WHY: Adds the project root folder to Python's list of places to look for modules.
# os.path.abspath(__file__) → full path to this file, e.g., D:\project\app.py
# os.path.dirname(...)      → strips the filename → D:\project\
# sys.path.insert(0, ...)   → puts that folder FIRST in the search list
# This ensures "from src.ocr_engine import extract_text" works correctly
# no matter where you run the script from.


# ─────────────────────────────────────────────────────
# SECTION 3: PAGE CONFIGURATION
# ─────────────────────────────────────────────────────

# IMPORTANT: This MUST be the very first Streamlit function called.
# Streamlit throws an error if any other st.* function runs before this.
st.set_page_config(
    page_title="MediAlt",          # Text shown in browser tab
    page_icon="icon.png",                      # Emoji shown as browser tab icon
    layout="wide",                      # "wide" uses full browser width (better for 2 columns)
                                        # alternative: "centered" (narrow single column)
    initial_sidebar_state="expanded",   # Sidebar is open when the app first loads
)


# ─────────────────────────────────────────────────────
# SECTION 4: CUSTOM CSS STYLING
# ─────────────────────────────────────────────────────

# st.markdown() injects raw HTML/CSS into the page.
# unsafe_allow_html=True is required to allow raw HTML — Streamlit blocks it by default
# for security reasons. Here we trust our own CSS so it is safe.
st.markdown("""
<style>

/* ── MAIN BACKGROUND ─────────────────────────────────────
   data-testid is Streamlit's internal identifier for the main content area.
   We override its background with a soft blue-gray color.
   !important forces this to override Streamlit's default styles. */
[data-testid="stAppViewContainer"] {
    background: #dde5f5 !important;
}

/* ── SIDEBAR BACKGROUND ─────────────────────────────────
   The sidebar gets a deep navy-to-blue gradient.
   linear-gradient(180deg, ...) means the gradient goes top-to-bottom. */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d47a1,#1565c0) !important;
}

/* Make ALL text inside the sidebar white so it reads against the dark background */
[data-testid="stSidebar"] * { color: white !important; }

/* Style the API key input field inside the dark sidebar */
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.15) !important;   /* semi-transparent white bg */
    color: white !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
}

/* Style the region dropdown (selectbox) in the sidebar */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
    color: white !important;
}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
    color: white !important;    /* selected value text */
}
[data-testid="stSidebar"] .stSelectbox svg {
    fill: white !important;     /* dropdown arrow icon */
}

/* ── TOP HEADER BANNER ──────────────────────────────────
   .rx-header is applied to our custom header div at the top of the page.
   gradient goes left-to-right (120deg angle) */
.rx-header {
    background: linear-gradient(120deg,#1565C0,#283593);
    color: white;
    padding: 1.8rem 2.5rem;   /* 1.8rem top/bottom, 2.5rem left/right */
    border-radius: 16px;       /* rounded corners */
    margin-bottom: 1.5rem;     /* space below the header */
}
.rx-header h1 { margin:0; font-size:2.2rem; color:white!important }
.rx-header p  { margin:0.2rem 0 0; opacity:0.85; color:white!important }

/* ── GENERIC WHITE CARD ─────────────────────────────────
   Used for the prescription details expander area */
.card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    border: 1px solid #dde3f0;
    box-shadow: 0 2px 8px rgba(21,101,192,0.08);   /* subtle blue shadow */
    margin-bottom: 1rem;
}

/* ── MEDICINE ROW CARD ───────────────────────────────────
   Each medicine gets its own blue panel with a left accent border */
.med-row {
    background: #e8f4fd;          /* light blue background */
    border-left: 4px solid #1976D2;  /* thick blue left border = accent */
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
}
.med-name { font-size:1.1rem; font-weight:700; color:#0d1b4b; }  /* drug name */
.med-meta { font-size:0.88rem; color:#2c3e6b; margin-top:0.3rem; }  /* dosage details */

/* ── ALTERNATIVE BRAND PILL ─────────────────────────────
   Green pill badges showing cheaper alternative brands */
.alt-pill {
    background: #1b5e20;       /* dark green */
    color: white;
    border-radius: 20px;       /* fully rounded = pill shape */
    padding: 3px 12px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-block;     /* allows multiple pills side by side */
    margin: 2px;
}

/* ── TAB NAVIGATION STYLES ──────────────────────────────
   Streamlit's tab component uses BaseWeb internally.
   These selectors target Streamlit's internal tab elements. */
.stTabs [data-baseweb="tab"] {
    color: #1565C0 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {   /* the currently active tab */
    background: #e3f2fd !important;
    border-bottom: 3px solid #1565C0 !important;
}

/* Force headings to dark navy so they are readable against the blue-gray background */
h1,h2,h3 { color: #0d1b4b !important; }

/* Make text input values dark (readable against white input background) */
.stTextInput input { color: #0d1b4b !important; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# SECTION 5: SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────

# CONCEPT: Streamlit reruns the entire script top-to-bottom on EVERY user interaction
# (button click, file upload, text input, etc.). This means variables reset each time.
# st.session_state is a special dictionary that PERSISTS across reruns.
# Think of it like a memory that survives page refreshes.

def _init():
    """
    Initialize session state variables with safe default values.
    Called once at startup. The 'if k not in st.session_state' check
    prevents resetting values that the user has already set.
    """
    defaults = {
        "ocr_result": None,    # Will hold the raw text string returned by Donut OCR
        "analysis":   None,    # Will hold the structured dict returned by Gemini
        "chat_history": [],    # Will hold a list of {"role":..., "content":...} dicts
        "image":      None,    # Will hold the PIL Image object uploaded by the user
        "analyzed":   False,   # Boolean flag: True once Gemini analysis completes
    }
    for k, v in defaults.items():
        if k not in st.session_state:   # Only set if not already in session
            st.session_state[k] = v

_init()   # Run the initializer immediately when the script loads


# ─────────────────────────────────────────────────────
# SECTION 6: SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────

# 'with st.sidebar:' is a context manager. Everything indented inside it
# appears in the left sidebar panel, not the main page.
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Text input for the Gemini API key.
    # type="password" masks the characters so the key is not visible on screen.
    # value=os.getenv(...) pre-fills the field with the key from the .env file.
    # The user can also type a key directly here — it overrides the .env value.
    gemini_key = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),  # "" is the fallback if .env is missing
        type="password",
        help="Get your key at https://aistudio.google.com/",  # tooltip text
    )

    st.markdown("---")   # Horizontal divider line
    st.markdown("### 🔧 Pipeline Options")

    # Checkbox to enable/disable the local Donut OCR step.
    # When checked (True): image goes through Donut first, then Gemini.
    # When unchecked (False): image goes directly to Gemini Vision only.
    # This is useful for testing or if the local model is not downloaded.
    use_donut = st.checkbox(
        "Local Donut OCR",
        value=True,   # checked by default
        help="Uses local Donut model. Uncheck to rely solely on Gemini Vision.",
    )

    st.markdown("---")
    st.markdown("### 🌍 Region")

    # Dropdown for selecting the country/region.
    # This value is passed to Gemini so it suggests brands available in that region.
    # index=0 means "India" is selected by default (index 0 in the list).
    region = st.selectbox(
        "Brand suggestions for",
        ["India", "USA", "UK", "Global"],
        index=0,
    )

    st.markdown("---")
    # st.caption() renders small gray text — good for disclaimers
    st.caption("⚠️ For research only. Not for clinical use.")
    st.caption("Model: Donut (NAVER Clova) + gemini-3-flash-preview")


# ─────────────────────────────────────────────────────
# SECTION 7: TOP HEADER BANNER
# ─────────────────────────────────────────────────────

# Render a custom HTML div using the .rx-header CSS class defined above.
# &amp; is the HTML entity for the & character (required inside HTML strings).
import base64

def get_base64(img):
    with open(img, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("icon.png")

st.markdown(f"""
<div style="display:flex; align-items:center; gap:15px;">
    <img src="data:image/png;base64,{img_base64}" width="100">
    <div>
        <h1>MediAlt</h1>
        <p>AI-powered Medical Prescription Analyzer & Alternative Brand Suggester</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# SECTION 8: THREE MAIN TABS
# ─────────────────────────────────────────────────────

# st.tabs() creates clickable tab buttons at the top of the content area.
# It returns one container object per tab.
# We unpack them into three variables: tab_analyze, tab_chat, tab_howto.
tab_analyze, tab_chat, tab_howto = st.tabs(
    ["📤 Upload & Analyze", "💬 Ask a Question", "📖 How It Works"]
)


# ═══════════════════════════════════════════════════════
# TAB 1: UPLOAD & ANALYZE
# ═══════════════════════════════════════════════════════

with tab_analyze:   # Everything inside this block appears under Tab 1

    # Split the tab into two columns:
    # col_left  = 1 part wide  (upload panel)
    # col_right = 1.7 parts wide (results panel — wider because it shows more)
    # gap="large" adds spacing between the two columns
    col_left, col_right = st.columns([1, 1.7], gap="large")

    # ── LEFT COLUMN: IMAGE UPLOAD ─────────────────────
    with col_left:
        st.markdown("### 📷 Upload Prescription")

        # File uploader widget.
        # type=[...] restricts accepted file formats.
        # label_visibility="collapsed" hides the label text above the widget
        # (we already printed a heading above it).
        uploaded = st.file_uploader(
            "Drag & drop or click",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        # Only run this block if the user has uploaded a file
        if uploaded:
            # Convert the uploaded file bytes into a PIL Image.
            # .convert("RGB") ensures 3-channel color (handles PNG transparency, grayscale, etc.)
            image = Image.open(uploaded).convert("RGB")

            # Save the image to session state so it persists across reruns
            st.session_state.image = image

            # Display the image in the UI
            st.image(image, caption="Uploaded Prescription", width="stretch")

            # Primary action button — styled blue/filled
            # use_container_width=True makes it stretch to fill the column width
            analyze_btn = st.button(
                "🔍 Analyze Prescription", type="primary", use_container_width=True
            )

            # Only run the analysis pipeline when the button is clicked
            if analyze_btn:

                # Guard: make sure the user has provided an API key before proceeding
                if not gemini_key:
                    st.error("🔑 Please enter your Gemini API key in the sidebar.")
                else:
                    # ── STEP 1: DONUT OCR ─────────────────────────────
                    # Start with empty string — if Donut is disabled or fails,
                    # Gemini will rely on the image alone
                    ocr_text = ""

                    if use_donut:   # Only run if the checkbox is checked
                        # st.spinner() shows a loading animation while the block runs
                        with st.spinner("🔍 Running Donut OCR…"):
                            try:
                                # Lazy import — only imported when needed.
                                # This avoids loading the 800MB model at startup.
                                from src.ocr_engine import extract_text

                                # Call Donut — takes ~2 seconds on CPU
                                # Returns a raw text string from the image
                                ocr_text = extract_text(image)

                                # Save raw OCR text to session state for the expander below
                                st.session_state.ocr_result = ocr_text

                            except FileNotFoundError:
                                # model/ folder is missing — user needs to download it
                                st.warning(
                                    "⚠️ Local model not found. "
                                    "Run `python model_download.py` or disable Donut OCR. "
                                    "Falling back to Gemini Vision only."
                                )
                            except Exception as e:
                                # Any other Donut error — gracefully fall back to Gemini only
                                st.warning(f"Donut OCR skipped ({e}). Using Gemini Vision only.")

                    # ── STEP 2: GEMINI AI ANALYSIS ────────────────────
                    with st.spinner("🤖 Analyzing with Gemini…"):
                        try:
                            from src.gemini_analyzer import analyze_prescription

                            # Send BOTH the image AND the OCR text to Gemini.
                            # Gemini uses both sources to cross-check and extract structure.
                            # ocr_text may be "" if Donut was skipped — Gemini handles that.
                            result = analyze_prescription(
                                image=image,
                                ocr_text=ocr_text,
                                api_key=gemini_key,
                                region=region,
                            )

                            # Store the structured result dict in session state
                            st.session_state.analysis = result

                            # Reset chat history when a new image is analyzed
                            # (so old conversation doesn't bleed into new prescription context)
                            st.session_state.chat_history = []

                            # Mark that analysis is complete — enables the chat tab
                            st.session_state.analyzed = True

                            st.success("✅ Analysis complete!")

                        except Exception as e:
                            # Show the actual error so the user knows what went wrong
                            st.error(f"Gemini error: {e}")

        else:
            # No file uploaded yet — show a placeholder message
            st.markdown("""
            <div class="empty-state">
              <div class="icon">📋</div>
              <h3>Upload a prescription image</h3>
              <p>Supports JPG, JPEG, PNG</p>
            </div>
            """, unsafe_allow_html=True)

    # ── RIGHT COLUMN: RESULTS DISPLAY ────────────────
    with col_right:

        # Retrieve the analysis result from session state.
        # This is None if no analysis has been done yet.
        data = st.session_state.analysis

        if data:
            # ── PRESCRIPTION HEADER INFO ───────────────
            # data["header"] contains patient name, doctor, date, etc.
            # .get("header", {}) safely returns {} if "header" key is missing
            header = data.get("header", {})

            # st.expander() creates a collapsible section.
            # expanded=True means it is open by default.
            with st.expander("🏥 Prescription Details", expanded=True):
                c1, c2 = st.columns(2)   # Two sub-columns for a clean layout

                # .get("patient_name", "N/A") → returns "N/A" if key is missing
                c1.markdown(f"**👤 Patient:** {header.get('patient_name', 'N/A')}")
                c1.markdown(f"**🎂 Age:** {header.get('age', 'N/A')}")
                c1.markdown(f"**🏥 Clinic:** {header.get('clinic', 'N/A')}")
                c2.markdown(f"**👨‍⚕️ Doctor:** {header.get('doctor_name', 'N/A')}")
                c2.markdown(f"**📅 Date:** {header.get('date', 'N/A')}")

                # Only show diagnosis if Gemini found one
                if header.get("diagnosis"):
                    st.markdown(
                        f"**🩺 Diagnosis:** "
                        f'<span class="badge badge-purple">{header["diagnosis"]}</span>',
                        unsafe_allow_html=True,
                    )

            # ── MEDICINES LIST ─────────────────────────
            st.markdown("### 💊 Prescribed Medicines")
            medicines = data.get("medicines", [])   # List of medicine dicts

            if medicines:
                # enumerate(medicines, 1) gives (1, med1), (2, med2), etc.
                # so we can number the medicines 1, 2, 3...
                for i, med in enumerate(medicines, 1):

                    # Safely extract each field — .get() returns "" if missing
                    name     = med.get("name", "Unknown")
                    strength = med.get("strength", "")
                    freq     = med.get("frequency", "")
                    duration = med.get("duration", "")
                    instruct = med.get("instructions", "")
                    generic  = med.get("generic_name", "")
                    alts     = med.get("alternatives", [])   # list of brand name strings

                    # Render a styled HTML card for each medicine.
                    # Python f-strings (f"...") embed variable values directly.
                    # The conditional expressions check if generic/alts exist before rendering.
                    st.markdown(f"""
                    <div class="med-row">
                      <div style="flex:1">
                        <div class="med-name">{i}. {name}
                          <span class="badge badge-blue">{strength}</span>
                        </div>
                        <div class="med-meta">
                          📅 <b>Frequency:</b> {freq} &nbsp;|&nbsp;
                          ⏱ <b>Duration:</b> {duration} &nbsp;|&nbsp;
                          🍽 <b>Instructions:</b> {instruct}
                        </div>
                        {"<div class='med-meta'>🔬 <b>Generic:</b> " + generic + "</div>" if generic else ""}
                        {"<div class='alt-wrap'>" + "".join(f'<span class=\"alt-pill\">💰 {a}</span>' for a in alts) + "</div>" if alts else ""}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Gemini returned an empty medicines list — image may not be a prescription
                st.info("No medicines extracted. The image may not be a prescription.")

            # ── WARNINGS / SPECIAL INSTRUCTIONS ───────
            warnings = data.get("warnings", [])   # List of warning strings
            if warnings:
                # Build an HTML <ul> list from the warnings list
                items = "".join(f"<li>{w}</li>" for w in warnings)
                st.markdown(
                    f'<div class="warning-box">⚠️ <b>Important Notes</b><ul>{items}</ul></div>',
                    unsafe_allow_html=True,
                )

        elif st.session_state.image is None:
            # No image uploaded yet — show placeholder
            st.markdown("""
            <div class="empty-state">
              <div class="icon">🤖</div>
              <h3>Results will appear here</h3>
              <p>Upload and analyze a prescription on the left</p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# TAB 2: Q&A CHAT
# ═══════════════════════════════════════════════════════

with tab_chat:

    # Guard: chat only makes sense after a prescription has been analyzed.
    # st.session_state.analyzed is set to True in Tab 1 after Gemini runs.
    if not st.session_state.analyzed:
        st.info("📤 Please analyze a prescription first in the **Upload & Analyze** tab.")
    else:
        st.markdown("### 💬 Ask About This Prescription")
        st.caption(
            "Examples: 'What are the side effects of these medicines?', "
            "'Can I take these with food?', 'Are there any drug interactions?'"
        )

        # Replay all previous messages from session state history.
        # This recreates the chat visually each time the page reruns.
        # Without this loop, the chat history would disappear on every rerun.
        for msg in st.session_state.chat_history:
            # st.chat_message() creates a styled message bubble.
            # "user" = right-aligned, "assistant" = left-aligned with bot icon.
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # st.chat_input() renders the text box at the bottom of the page.
        # The walrus operator (:=) assigns AND checks the value in one line.
        # "if question := ..." means: capture the input, and only proceed if not empty.
        if question := st.chat_input("Ask a question about the prescription…"):

            # Add user's question to chat history for replay next rerun
            st.session_state.chat_history.append({"role": "user", "content": question})

            # Show the user's message bubble immediately
            with st.chat_message("user"):
                st.markdown(question)

            # Show assistant's response bubble
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        from src.gemini_analyzer import chat_about_prescription

                        # Send the full conversation history to Gemini.
                        # history[:-1] = all messages EXCEPT the current question
                        # (the current question is passed separately as 'question')
                        answer = chat_about_prescription(
                            analysis=st.session_state.analysis,  # the prescription JSON context
                            question=question,                     # current user question
                            history=st.session_state.chat_history[:-1],  # previous turns
                            api_key=gemini_key,
                        )
                    except Exception as e:
                        answer = f"❌ Error: {e}"

                st.markdown(answer)

            # Add assistant's reply to history so it appears on next rerun
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )


# ═══════════════════════════════════════════════════════
# TAB 3: HOW IT WORKS
# ═══════════════════════════════════════════════════════

with tab_howto:
    st.markdown("## 🔄 System Workflow")

    # A list of tuples: (emoji_icon, step_title, description_text)
    # Used to programmatically generate the workflow steps instead of copy-pasting HTML
    steps = [
        ("📤", "Upload",           "User uploads a JPG/PNG of a handwritten or printed prescription."),
        ("🔍", "Donut OCR",        "Fine-tuned NAVER Clova Donut model transcribes raw text from the image."),
        ("🤖", "Gemini Cross-Verify", "Gemini re-reads the image, corrects OCR errors, and structures the data into JSON."),
        ("💊", "Brand Suggestions","Gemini identifies the generic drug name and lists affordable alternative brands."),
        ("💬", "Interactive Q&A",  "Ask follow-up questions — drug interactions, dosage clarifications, side effects."),
    ]

    # enumerate(steps, 1) starts counting from 1 (not 0) for step numbers
    for i, (icon, title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-row">
          <div class="step-num">{i}</div>
          <div class="card" style="margin:0;flex:1;padding:0.8rem 1.2rem">
            {icon} <b>{title}</b> — {desc}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🏗️ Tech Stack")

    # Two-column layout for the tech stack tables
    col1, col2 = st.columns(2)
    with col1:
        # Streamlit renders Markdown tables automatically
        st.markdown("""
        | Component | Technology |
        |-----------|-----------|
        | OCR Model | Donut (NAVER Clova) |
        | AI Analysis | Google Gemini Flash |
        | UI | Streamlit |
        """)
    with col2:
        st.markdown("""
        | Component | Technology |
        |-----------|-----------|
        | Training | PyTorch Lightning |
        | Dataset | HuggingFace Datasets |
        | Hosting | HuggingFace Hub |
        """)

    st.markdown("---")
    # st.warning() displays a yellow warning box
    st.warning(
        "⚠️ **Disclaimer:** This tool is for research and educational purposes only. "
        "It is **not validated for clinical use**. Do not make medical decisions based on its output."
    )
