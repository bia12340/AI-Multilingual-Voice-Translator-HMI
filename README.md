# AI-Driven Multilingual Voice Translator: An HMI Use Case
## Final Project for the Human-Machine Interface (HMI) Course

### Project Context
This project was developed as a **collaborative team effort** for the final evaluation of the **Human-Machine Interface (HMI)** course. The application represents an advanced study into integrating neural processing pipelines with modern graphical interfaces, focusing on the seamless interaction between human users and Artificial Intelligence systems.

---

### Project Overview
The project focuses on the implementation of an **AI-Driven Voice Translation** system, specifically exploring the capabilities of **Speech-to-Text (STT)**, **Neural Machine Translation (NMT)**, and **Speech Synthesis (TTS)**. We analyzed how a modular software architecture can support real-time linguistic flexibility—supporting 5 international languages—while maintaining high responsiveness and accuracy through quantized neural models.

---

### Key Technical Contributions
* **Neural Pipeline Architecture:** Integration of a cascading translation logic using English as a "Pivot Language" to ensure high-fidelity results across diverse language pairs.
* **Transcription Optimization:** Implementation of Faster-Whisper with **Voice Activity Detection (VAD)** and **Beam Search** to minimize background noise and maximize grammatical accuracy.
* **Asynchronous Orchestration:** Development of a multi-threaded execution environment to ensure the GUI remains fluid (60 FPS) during heavy AI model inference.
* **UI/UX Adaptive Design:** Design of a responsive interface using **CustomTkinter**, featuring dynamic visual feedback, session history persistence, and real-time theme toggling.

---

### Technologies & Frameworks
* **Models:** Faster-Whisper (STT), Helsinki-NLP MarianMT (NMT), Edge-TTS (Neural Synthesis).
* **Core Libraries:** Python, CustomTkinter (GUI), Asyncio & Threading (Parallelism), FFmpeg (Audio Management).
* **Focus Areas:** Natural Language Processing (NLP), Signal Processing, and Human-Computer Interaction (HCI).

---

### Comment regarding the documentation
The source code available on GitHub has been updated with English documentation and comments for international accessibility, while the snippets presented in this document reflect the initial development phase.
