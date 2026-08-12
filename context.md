# Project

Semester research project.

Goal:
Combine findings from two research papers.

Paper 1:
Lightweight Vision Language Model Guided Gesture Recognition Based on EMG.

Repository:
VLM-EMG4Gesture

Current objective:

- Understand architecture
- Verify repository works
- Verify datasets exist
- Understand preprocessing
- Understand EMG decoder
- Understand VLM pipeline
- Later integrate second research paper.

Current understanding:

Architecture:

Image
→ Vision Encoder
→ Linear Projection
→ Language Model

Prompt
→ Tokenizer
→ Language Model

EMG
→ EMG Decoder

Outputs:

- Grasp / Release / Neutral
- Object Type
- Gesture Type

The VLM verifies EMG predictions using pseudolabels.

Whenever explaining code:

- Relate every function back to the research paper.
- Mention which figure or algorithm it implements.
- Explain mathematically where needed.
