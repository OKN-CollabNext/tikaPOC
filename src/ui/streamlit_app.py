# File: src/ui/streamlit_app.py

import sys
import os

# Append the parent directory (i.e. "src") to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(os.path.abspath(parent_dir))

import streamlit as st
from services.chat_agent import ChatManager
from PIL import Image

# Import your segmentation service
from segmentation.segmentation_agent import SegmentationAgent

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("Socratic RAG Agent with Image Segmentation")

    # Initialize session state for chat
    initialize_session_state()

    # --- Chat Section (existing code) ---
    st.header("Research Topics Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What research topics are you interested in?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.session_state.chat_manager.handle_message(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # --- New Image Segmentation Section ---
    st.header("Image Segmentation Demo")
    uploaded_file = st.file_uploader("Upload an image for segmentation", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Segment Image"):
            seg_agent = SegmentationAgent()
            segmented_mask = seg_agent.segment_image(image)
            st.image(segmented_mask, caption="Segmentation Mask", use_column_width=True)

    # Reset button for chat conversation
    with st.sidebar:
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chat_manager.reset_conversation()
            st.rerun()

if __name__ == "__main__":
    main()
