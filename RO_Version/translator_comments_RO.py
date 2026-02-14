"""
PROIECT: Translator Vocal Multilingv (Interfață Om-Mașină)
DESCRIERE: Aplicație pentru traducere vocală în timp real folosind modele de AI
TEHNOLOGII: Faster-Whisper (STT), MarianMT (NMT), Edge-TTS (TTS), CustomTkinter (GUI)
AUTORI: Oteșanu Alexandra Maria, Raicu Bianca Elena, Rusen Andreea Emela
"""

import customtkinter as ctk
from tkinter import StringVar, IntVar, BooleanVar, messagebox
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import asyncio
from dataclasses import dataclass
from datetime import datetime
import edge_tts
import sounddevice as sd
import numpy as np
import os
import tempfile
import winsound
import soundfile as sf
import subprocess
import threading
import requests

# --------------------------------------------
# Configurare interfață grafică CustomTkinter
# --------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# -----------------------------------
# Configurări TTS și setări generale
# -----------------------------------
DEBUG = False
TARGET_LANGUAGE = "en"

# Vocile Edge-TTS pentru limbile suportate
EDGE_VOICES = {
    'es': 'es-ES-AlvaroNeural',
    'en': 'en-US-GuyNeural',
    'fr': 'fr-FR-HenriNeural',
    'it': 'it-IT-DiegoNeural',
    'ro': 'ro-RO-AlinaNeural'
}

# -----------------------------
# Gestionare Istoric și Replay
# -----------------------------

# Structură de date pentru stocarea traducerilor anterioare în memoria sesiunii
@dataclass
class HistoryItem:
    ts: str
    src_lang: str
    tgt_lang: str
    original: str
    translated: str
    voice: str

history: list[HistoryItem] = []

#Salvează o nouă intrare în istoric și adaugă ora exactă a procesării
def add_to_history(src_lang: str, tgt_lang: str, original: str, translated: str, voice: str):
    history.append(
        HistoryItem(
            ts=datetime.now().strftime("%H:%M:%S"),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            original=original,
            translated=translated,
            voice=voice
        )
    )

# Funcție de acces rapid pentru a re-asculta ultima traducere efectuată
def replay_last():
    if not history:
        return
    item = history[-1]
    edge_speak_blocking(item.translated, item.voice)

# --------------------
# Inițializare modele
# --------------------

# Folosim modelele MarianMT de la Helsinki-NLP pentru traducere
MODEL_MUL_EN = "Helsinki-NLP/opus-mt-mul-en"
MODEL_EN_MUL = "Helsinki-NLP/opus-mt-en-mul"

EN_MUL_TARGET_TOKEN = {
    "it": ">>ita<<",
    "fr": ">>fra<<",
    "es": ">>spa<<",
    "ro": ">>ron<<",
}

tok_mul_en = mod_mul_en = tok_en_mul = mod_en_mul = whisper_model = None

#Încărcarea modelelor în memoria RAM/VRAM la pornirea aplicației
def load_models(mode="accurate"):
    global tok_mul_en, mod_mul_en, tok_en_mul, mod_en_mul, whisper_model
    
    tok_mul_en = MarianTokenizer.from_pretrained(MODEL_MUL_EN)
    mod_mul_en = MarianMTModel.from_pretrained(MODEL_MUL_EN)
    tok_en_mul = MarianTokenizer.from_pretrained(MODEL_EN_MUL)
    mod_en_mul = MarianMTModel.from_pretrained(MODEL_EN_MUL)
    
    # Setarea modelelor în modul de evaluare
    mod_mul_en.eval()
    mod_en_mul.eval()

    # Selecția dimensiunii modelului Whisper pentru a alege mai ușor dintre acuratețe și viteză
    whisper_model_name = "tiny" if mode == "fast" else "small"
    whisper_model = WhisperModel(
        whisper_model_name,
        device="cpu", # Se poate schimba în "cuda" pentru accelerare GPU
        compute_type="int8" #Optimizare pentru consum redus de memorie
    )

# ---------------------------------------
# Funcții de procesare text și traducere
# ---------------------------------------

#Returnează textul recunoscut și limba detectată
def transcribe_and_detect(audio_file):
    segments, info = whisper_model.transcribe(
        audio_file,
        beam_size=5, #Analiză probabilistică pentru creșterea acurateței
        vad_filter=True # Filtru pentru eliminarea perioadelor de liniște
    )
    detected_lang = (info.language or "unknown").lower()
    text = "".join(seg.text for seg in segments).strip()
    return detected_lang, text

# Traducere din limba sursă în limba engleză
def _translate_mul_to_en(text: str, detected_lang: str) -> str:
    inputs = tok_mul_en([text], return_tensors="pt", padding=True)
    
    # Generarea traducerii cu parametri de control pentru eliminarea erorilor de vocabular
    out = mod_mul_en.generate(
        **inputs, 
        max_length=256, 
        num_beams=5, # Analizează 5 variante paralele pentru a alege cea mai coerentă frază
        repetition_penalty=2.5, # Împiedică traducerile repetitive sau proaste 
        no_repeat_ngram_size=2, # Interzice repetarea aceleiași succesiuni de 2 cuvinte
        early_stopping=True
    )
    return tok_mul_en.decode(out[0], skip_special_tokens=True)

# Traducere din limba engleză în limba dorită
def _translate_en_to_mul(text: str, target_lang: str) -> str:
    token = EN_MUL_TARGET_TOKEN.get(target_lang)
    if not token:
        return f"[Translation unavailable: en->{target_lang}]"

    # Adăugarea token-ului special pentru a ghida modelul către limba corectă
    text_with_token = f"{token} {text}"
    inputs = tok_en_mul([text_with_token], return_tensors="pt", padding=True)
    out = mod_en_mul.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
    return tok_en_mul.decode(out[0], skip_special_tokens=True)

# Traducerea textului din limba sursă în limba țintă
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if not text:
        return ""

    # Dicționar pentru denumirile complete ale câtorva limbi uzuale
    language_names = {
        'en': 'English',
        'ro': 'Romanian',
        'fr': 'French',
        'it': 'Italian',
        'es': 'Spanish',
        'zh': 'Chinese/Mandarin',
        'de': 'German',
        'ja': 'Japanese',
        'ru': 'Russian',
        'pt': 'Portuguese',
        'ar': 'Arabic', 
        'hi': 'Hindi',
        'tr': 'Turkish',
        'ko': 'Korean',
    }

    source_lang = source_lang.lower().strip()
    target_lang = target_lang.lower().strip()

    # Normalizăm codurile lingvistice pentru MarianMT 
    lang_map = {
        'ron': 'ro', 
        'md': 'ro', 
        'moldavian': 'ro',
        'ro': 'ro',
        'fra': 'fr',
        'ita': 'it',
        'spa': 'es'
    }
    source_lang = lang_map.get(source_lang.lower().strip(), source_lang.lower().strip())

    # Dacă limba detectată nu este una dintre limbile suportate, returnăm un mesaj de notificare
    supported_sources = ['ro', 'en', 'fr', 'it', 'es']
    if source_lang not in supported_sources:
    # semnalăm că e nevoie de fallback API
        return None


    if source_lang == target_lang:
        return text
    
    # Dacă limba engleză nu este limba sursă, traducem în engleză
    en_text = text if source_lang == "en" else _translate_mul_to_en(text, source_lang)

    # Dacă limba engleză nu este limba țintă, traducem în limba țintă
    if target_lang == "en":
        return en_text

    return _translate_en_to_mul(en_text, target_lang)


def translate_via_api(text: str, source_lang: str, target_lang: str):
    try:
        # Folosim Google Translate API
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": target_lang,
            "dt": "t",
            "q": text
        }
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Extragem textul din structura [[["traducere", "original", ...]]]
        data = response.json()
        if data and data[0]:
            # Uneori textul lung e spart în mai multe segmente
            full_translation = "".join([segment[0] for segment in data[0] if segment[0]])
            return full_translation
        return None
    except Exception as e:
        print(f"Translation API failed: {e}")
        return None



# -------------------
# TTS via Edge-TTS
# -------------------

# Sinteză vocală asincronă: transformă textul în fișier audio MP3/WAV
async def _edge_synthesize_to_wav(text, voice, out_path, rate=0, volume=0):
    rate_str = f"{rate:+d}%"
    volume_str = f"{volume:+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str, volume=volume_str)
    await communicate.save(out_path)

# Gestionează redarea audio și include un sistem de 'fallback' cu FFmpeg
def edge_speak_blocking(text, voice, rate=0, volume=0):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp_name = tmp.name
    tmp.close()
    fs_local = 16000 # Frecvență eșantionare standard
    try:
        try:
            asyncio.run(_edge_synthesize_to_wav(text, voice, tmp_name, rate, volume))
        except RuntimeError:
            # Creare loop nou dacă cel principal este blocat (pentru stabilitate UI)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(edge_tts.Communicate(text, voice).save(tmp_name))
            loop.close()

        # Verificarea formatului fișierului pentru redare nativă sau conversie
        with open(tmp_name, 'rb') as f:
            header = f.read(4)

        if header.startswith(b'RIFF'): # Format WAV valid
            if os.name == 'nt':
                winsound.PlaySound(tmp_name, winsound.SND_FILENAME)
            else:
                data, sr = sf.read(tmp_name)
                sd.play(data, sr)
                sd.wait()
        else:
            # Conversie forțată via FFmpeg dacă fișierul este comprimat sau corupt
            conv_tmp = tmp_name + '.conv.wav'
            try:
                cmd = ['ffmpeg', '-y', '-i', tmp_name, '-ar', str(fs_local), '-ac', '1', conv_tmp]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.name == 'nt':
                    winsound.PlaySound(conv_tmp, winsound.SND_FILENAME)
                else:
                    data, sr = sf.read(conv_tmp)
                    sd.play(data, sr)
                    sd.wait()
            except Exception as e:
                if DEBUG:
                    print('[edge-tts] ffmpeg conversion/playback failed:', e)
            finally:
                try:
                    # Ștergerea fișierului temporar pentru a economisi spațiu pe disc
                    if os.path.exists(conv_tmp):
                        os.remove(conv_tmp)
                except Exception:
                    pass
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass

# -------------------
# Captura audio în timp real
# -------------------
fs = 16000
recording = False
audio_buffer = []

# Funcție apelată în timp real de placa de sunet pentru a umple buffer-ul audio
def audio_callback(indata, frames, time, status):
    if recording:
        audio_buffer.append(indata.copy())

# Pregătește buffer-ul și activează flag-ul de înregistrare
def start_recording():
    global recording, audio_buffer
    audio_buffer = []
    recording = True

def stop_recording():
    global recording
    recording = False

# Unește fragmentele audio capturate într-un singur fișier WAV
def save_recording():
    audio = np.concatenate(audio_buffer, axis=0)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_wav.name, audio, fs)
    return temp_wav.name

stream = sd.InputStream(
    samplerate=fs,
    channels=1,
    callback=audio_callback
)
stream.start()

# ---------------------------------
# Aplicația principală CustomTkinter
# ---------------------------------

class SpeechTranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Speech Translator")
        self.geometry("1100x700")
        self.minsize(1000, 650)

        # Variabile de stare
        self.mode_var = StringVar(value="accurate")
        self.target_var = StringVar(value="en")
        self.rate_var = IntVar(value=0)
        self.volume_var = IntVar(value=0)
        self.autoplay_var = BooleanVar(value=True)
        self.autocopy_var = BooleanVar(value=False)
        
        self.detected_lang_var = StringVar(value="-")
        self.status_var = StringVar(value="Idle")
        self.appearance_mode = StringVar(value="dark")
        
        self.settings_win = None

        # Aspect
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_controls()
        self._build_main_content()

        # Încarcă modelele
        self.set_status("Loading models...")
        self.update()
        try:
            load_models(self.mode_var.get())
            self.set_status("Idle")
        except Exception as e:
            messagebox.showerror("Error loading models", str(e))
            self.set_status("Error")

    def _build_header(self):
        # Antet cu titlu și status
        header = ctk.CTkFrame(self, corner_radius=12)
        header.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        # Titlul aplicației
        title = ctk.CTkLabel(header, text="🎙️ Speech Translator", 
                            font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=16, sticky="w")

        # Informații status
        status_frame = ctk.CTkFrame(header, fg_color="transparent")
        status_frame.grid(row=0, column=1, padx=20, sticky="e")
        
        ctk.CTkLabel(status_frame, text="Status:", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(0, 8))
        self.lbl_status = ctk.CTkLabel(status_frame, textvariable=self.status_var,
                                       font=ctk.CTkFont(size=12))
        self.lbl_status.pack(side="left")
        
        ctk.CTkLabel(status_frame, text=" | Detected:", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(20, 8))
        ctk.CTkLabel(status_frame, textvariable=self.detected_lang_var,
                    font=ctk.CTkFont(size=12)).pack(side="left")

    def _build_controls(self):
        # Bara de control cu butoane și setări
        controls = ctk.CTkFrame(self, corner_radius=12)
        controls.grid(row=1, column=0, padx=16, pady=8, sticky="ew")
        controls.grid_columnconfigure(5, weight=1)

        # Limba țintă
        ctk.CTkLabel(controls, text="Target:", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=(16, 8), pady=12)
            
        self.btn_lang_selector = ctk.CTkButton(controls, textvariable=self.target_var, 
                                               command=self.open_language_popup,
                                               width=60, fg_color="#1f538d")
        self.btn_lang_selector.grid(row=0, column=1, padx=8, pady=12)

        # Butoane acțiune principale
        self.btn_start = ctk.CTkButton(controls, text="🎙️ Start Recording", 
                                       command=self.on_start, width=140)
        self.btn_start.grid(row=0, column=2, padx=8, pady=12)

        self.btn_stop = ctk.CTkButton(controls, text="⏹️ Stop & Translate", 
                                      command=self.on_stop, state="disabled", width=140)
        self.btn_stop.grid(row=0, column=3, padx=8, pady=12)

        # Butoane utilitare
        ctk.CTkButton(controls, text="🔄 Replay Last", command=self.on_replay_last,
                     fg_color=("gray90", "gray20"),
                     hover_color=("gray85", "gray30"),
                     text_color=("black", "white"),
                     border_color=("gray50", "gray60"),
                     border_width=2,
                     width=120).grid(
            row=0, column=4, padx=8, pady=12)

        ctk.CTkButton(controls, text="⚙️ Settings", command=self.open_settings_popup,
                     fg_color=("gray90", "gray20"),
                     hover_color=("gray85", "gray30"),
                     text_color=("black", "white"),
                     border_color=("gray50", "gray60"),
                     border_width=2,
                     width=100).grid(
            row=0, column=6, padx=8, pady=12)
        
        # Buton schimbare temă
        self.theme_btn = ctk.CTkButton(controls, text="🌙", command=self.toggle_theme,
                                       fg_color=("gray90", "gray20"),
                                       hover_color=("gray85", "gray30"),
                                       text_color=("black", "white"),
                                       border_color=("gray50", "gray60"),
                                       border_width=2,
                                       width=50)
        self.theme_btn.grid(row=0, column=7, padx=(0, 16), pady=12)

    def _build_main_content(self):
        # Zona principală cu text și istoric
        main = ctk.CTkFrame(self, corner_radius=12)
        main.grid(row=2, column=0, padx=16, pady=(8, 16), sticky="nsew")
        main.grid_columnconfigure(0, weight=2)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # Partea stângă - Zone text
        left = ctk.CTkFrame(main, corner_radius=12)
        left.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        left.grid_rowconfigure(3, weight=1)

        # Text detectat
        ctk.CTkLabel(left, text="Detected Text", 
                    font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=14, pady=(14, 8), sticky="w")
        
        self.txt_detected = ctk.CTkTextbox(left, wrap="word", 
                                          font=ctk.CTkFont(size=13))
        self.txt_detected.grid(row=1, column=0, padx=14, pady=(0, 12), sticky="nsew")

        # Rezultat tradus
        ctk.CTkLabel(left, text="Translated Output", 
                    font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=2, column=0, padx=14, pady=(8, 8), sticky="w")
        
        self.txt_output = ctk.CTkTextbox(left, wrap="word",
                                        font=ctk.CTkFont(size=13))
        self.txt_output.grid(row=3, column=0, padx=14, pady=(0, 14), sticky="nsew")

        # Partea dreaptă - Istoric
        right = ctk.CTkFrame(main, corner_radius=12)
        right.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(right, text="History", 
                    font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=14, pady=(14, 8), sticky="w")

        # Listă istoric (folosind cadru derulabil pentru elemente selectabile)
        self.history_frame = ctk.CTkScrollableFrame(right, label_text="")
        self.history_frame.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="nsew")
        self.history_frame.grid_columnconfigure(0, weight=1)
        
        self.history_buttons = []
        self.selected_history_idx = None

        # History buttons
        btn_frame = ctk.CTkFrame(right, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(btn_frame, text="▶️ Replay Selected", 
                     command=self.on_replay_selected,
                     fg_color=("gray90", "gray20"),
                     hover_color=("gray85", "gray30"),
                     text_color=("black", "white"),
                     border_color=("gray50", "gray60"),
                     border_width=2).grid(
            row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(btn_frame, text="🗑️ Clear History", 
                     command=self.on_clear_history,
                     fg_color=("#ffdddd", "#8b0000"),      # light / dark
                     hover_color=("#ffcccc", "#a40000"),
                     text_color=("black", "white"),
                     border_color=("#cc0000", "#ff6666"),
                     border_width=2).grid(
            row=0, column=1, padx=(4, 0), sticky="ew")

    def open_settings_popup(self):
        # Fereastră popup pentru setări
        if self.settings_win and self.settings_win.winfo_exists():
            self.settings_win.focus()
            self.settings_win.lift()
            return

        self.settings_win = ctk.CTkToplevel(self)
        self.settings_win.title("Translation Settings")
        self.settings_win.geometry("480x380")
        self.settings_win.resizable(False, False)
        self.settings_win.transient(self)

        frame = ctk.CTkFrame(self.settings_win, corner_radius=12)
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        # Speech rate
        ctk.CTkLabel(frame, text="Speech Rate (%)", 
                    font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=16, pady=(16, 6))
        rate_slider = ctk.CTkSlider(frame, from_=-50, to=50, 
                                   number_of_steps=100, variable=self.rate_var)
        rate_slider.pack(fill="x", padx=16)
        self.rate_label = ctk.CTkLabel(frame, text=f"{self.rate_var.get()}%")
        self.rate_label.pack(anchor="w", padx=16, pady=(4, 0))
        
        def update_rate_label(*args):
            self.rate_label.configure(text=f"{self.rate_var.get()}%")
        self.rate_var.trace_add("write", update_rate_label)

        # Volume
        ctk.CTkLabel(frame, text="Volume (%)", 
                    font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=16, pady=(16, 6))
        volume_slider = ctk.CTkSlider(frame, from_=-50, to=50, 
                                     number_of_steps=100, variable=self.volume_var)
        volume_slider.pack(fill="x", padx=16)
        self.volume_label = ctk.CTkLabel(frame, text=f"{self.volume_var.get()}%")
        self.volume_label.pack(anchor="w", padx=16, pady=(4, 0))
        
        def update_volume_label(*args):
            self.volume_label.configure(text=f"{self.volume_var.get()}%")
        self.volume_var.trace_add("write", update_volume_label)

        # Checkboxes
        ctk.CTkCheckBox(frame, text="Auto-play translation", 
                       variable=self.autoplay_var,
                       font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=16, pady=(20, 8))
        
        ctk.CTkCheckBox(frame, text="Auto-copy translated text to clipboard", 
                       variable=self.autocopy_var,
                       font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=16, pady=(0, 16))

        # Close button
        ctk.CTkButton(frame, text="Close",
                     fg_color=("gray90", "gray20"),
                     hover_color=("gray85", "gray30"),
                     text_color=("black", "white"),
                     border_color=("gray50", "gray60"),
                     border_width=2,
                     command=self.settings_win.destroy).pack(
            anchor="e", padx=16, pady=(8, 16))
    
    def open_language_popup(self):
        # Creează o fereastră pop-up pentru alegerea limbii țintă
        popup = ctk.CTkToplevel(self)
        popup.title("Select Language")
        popup.geometry("250x350")
        popup.attributes("-topmost", True)  # Să rămână deasupra ferestrei principale
        popup.resizable(False, False)
        
        # Blocăm interacțiunea cu fereastra principală până se face alegerea
        popup.after(10, lambda: popup.grab_set()) 

        ctk.CTkLabel(popup, text="Choose Target Language", 
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=15)

        # Limbile suportate local
        supported_langs = [
            ("English", "en"),
            ("Spanish", "es"),
            ("Italian", "it"),
            ("French", "fr"),
            ("Romanian", "ro")
        ]

        def select_and_close(lang_code):
            self.target_var.set(lang_code)
            popup.destroy()

        # Creăm butoane pentru fiecare limbă
        for name, code in supported_langs:
            ctk.CTkButton(popup, text=name, 
                          command=lambda c=code: select_and_close(c),
                          width=180, height=35,
                          fg_color=("gray90", "gray20"),
                          hover_color=("gray85", "gray30"),
                          text_color=("black", "white"),
                          border_color=("gray50", "gray60"),
                          border_width=2).pack(pady=5)
            
        # Buton de închidere opțional
        ctk.CTkButton(popup, text="Cancel", command=popup.destroy, 
                      fg_color=("gray90", "gray20"),
                      hover_color=("gray85", "gray30"),
                      text_color=("black", "white"),
                      border_color=("gray50", "gray60"),
                      border_width=2,
                      width=100).pack(pady=15)

    def set_status(self, status: str):
        # Actualizează status-ul aplicației
        self.status_var.set(status)

    def ui_set_text(self, widget: ctk.CTkTextbox, value: str):
        # Setează text în textbox
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    def refresh_history(self):
        # Actualizează afișarea istoricului
        # Șterge butoanele existente
        for btn in self.history_buttons:
            btn.destroy()
        self.history_buttons = []
        self.selected_history_idx = None
        
        # Creează butoane noi pentru fiecare element din istoric
        for idx, item in enumerate(history):
            item_frame = ctk.CTkFrame(self.history_frame, corner_radius=8)
            item_frame.grid(row=idx, column=0, pady=4, padx=4, sticky="ew")
            item_frame.grid_columnconfigure(0, weight=1)
            
            # Item button
            text = f"[{item.ts}] {item.src_lang}→{item.tgt_lang}\n{item.translated[:60]}..."
            btn = ctk.CTkButton(
                item_frame, 
                text=text,
                command=lambda i=idx: self.select_history_item(i),
                fg_color=("gray90", "gray20"),
                border_color=("gray50", "gray60"),
                border_width=1,
                text_color=("black", "white"),
                hover_color=("gray70", "gray30"),
                anchor="w",
                height=60,
                font=ctk.CTkFont(size=11)
            )
            btn.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
            self.history_buttons.append(btn)
    
    def select_history_item(self, idx):
        # Selectează un item din istoric
        # Deselectează precedentul
        if self.selected_history_idx is not None and self.selected_history_idx < len(self.history_buttons):
            self.history_buttons[self.selected_history_idx].configure(fg_color=("gray90", "gray20"))
        
        # Selectează noul
        self.selected_history_idx = idx
        if idx < len(self.history_buttons):
            self.history_buttons[idx].configure(fg_color=("#cce4ff", "#2b4f81"))

    def on_start(self):
        # Începe înregistrarea
        if recording:
            return
        start_recording()
        self.set_status("Recording...")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

    def on_stop(self):
        # Oprește înregistrarea și procesează
        if not recording:
            return
        self.set_status("Processing...")
        self.btn_stop.configure(state="disabled")
        threading.Thread(target=self._process_stop_worker, daemon=True).start()

    def _process_stop_worker(self):
        # Fir de execuție worker pentru procesare
        global recording
        stop_recording()

        if not audio_buffer:
            self.after(0, lambda: self.set_status("Idle"))
            self.after(0, lambda: self.btn_start.configure(state="normal"))
            self.after(0, lambda: self.btn_stop.configure(state="disabled"))
            return

        audio_file = save_recording()

        try:
            # Transcriere
            detected_lang, text = transcribe_and_detect(audio_file)

            # Traducere
            tgt = self.target_var.get().strip().lower()
            used_fallback = False
            api_result = None
            if detected_lang == tgt:
                translated_text = text
            else:
                translated_text = translate_text(text, detected_lang, tgt)
                used_fallback = False
                api_result = None

                if translated_text is None:
                    api_tgt = tgt.split("-")[0]
                    api_result = translate_via_api(text,"auto", tgt)
                    print("API RESULT:", repr(api_result))
                    used_fallback = True

            voice = EDGE_VOICES.get(tgt, EDGE_VOICES['en'])



            
            # Actualizează interfața
            def _ui_done(translated_text, detected_lang, text, used_fallback, api_result):
                self.detected_lang_var.set(detected_lang)
                self.ui_set_text(self.txt_detected, text)

                # Inițializăm cu ce avem de la modelele locale
                final_translation = translated_text

                if used_fallback:
                    # Întrebăm utilizatorul
                    use_api = messagebox.askyesno(
                        "Online translation",
                        f"Detected language: {detected_lang}.\nLocal translation not supported.\n\n"
                        "Do you want to use the online API result?"
                    )

                    if use_api:
                        if api_result: # Aici verificăm ce a descărcat API-ul în fundal
                            final_translation = api_result
                        else:
                            final_translation = "[Error: API returned empty result]"
                    else:
                        final_translation = "[Translation cancelled by user]"

                # Afișăm în interfață
                self.ui_set_text(self.txt_output, final_translation)
                self.set_status("Idle")
                self.btn_start.configure(state="normal")
                self.btn_stop.configure(state="disabled")

                # Adăugăm în istoric și vorbim DOAR dacă avem text valid
                if final_translation and not final_translation.startswith("["):
                    add_to_history(detected_lang, tgt, text, final_translation, voice)
                    self.refresh_history()

                    # Copiere automată în clipboard dacă e activată
                    if self.autocopy_var.get():
                        self.clipboard_clear()
                        self.clipboard_append(final_translation)

                    if self.autoplay_var.get():
                        threading.Thread(
                            target=lambda: edge_speak_blocking(
                                final_translation,
                                voice,
                                self.rate_var.get(),
                                self.volume_var.get()
                            ),
                            daemon=True
                        ).start()


            self.after(0, lambda: _ui_done(translated_text, detected_lang, text, used_fallback, api_result))

        finally:
            try:
                os.remove(audio_file)
            except Exception:
                pass

    def on_replay_last(self):
        # Redă ultima traducere
        if not history:
            return
        item = history[-1]
        threading.Thread(
            target=lambda: edge_speak_blocking(item.translated, item.voice, 
                                              self.rate_var.get(), self.volume_var.get()), 
            daemon=True
        ).start()
    
    def on_replay_selected(self):
        # Redă traducerea selectată din istoric
        if self.selected_history_idx is None:
            messagebox.showinfo("No Selection", "Please select a history item first.")
            return
        
        if self.selected_history_idx >= len(history):
            return
        
        item = history[self.selected_history_idx]
        threading.Thread(
            target=lambda: edge_speak_blocking(item.translated, item.voice, 
                                              self.rate_var.get(), self.volume_var.get()), 
            daemon=True
        ).start()

    def on_clear_history(self):
        # Șterge istoricul
        history.clear()
        self.refresh_history()
    
    def toggle_theme(self):
        # Schimbă între modul light și dark
        current = self.appearance_mode.get()
        if current == "dark":
            ctk.set_appearance_mode("light")
            self.appearance_mode.set("light")
            self.theme_btn.configure(text="☀️")
        else:
            ctk.set_appearance_mode("dark")
            self.appearance_mode.set("dark")
            self.theme_btn.configure(text="🌙")

    def on_close(self):
        # Închide aplicația
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = SpeechTranslatorApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()