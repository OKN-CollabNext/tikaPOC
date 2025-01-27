import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from services.chat_agent import ChatManager

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # This is critical. We need to add in the place to remember what is the
    # final and last input of the user for the generation of JSON
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""

def main():
    st.title("Socratic RAG Agent")

    # Initialize session state
    initialize_session_state()

    # 1) Add a text input dedicated to autocomplete
    #    Streamlit re-runs your script every time the user types by default.
    partial_input = st.text_input("Type a keyword to auto-complete")

    # 2) Query the autocomplete method
    if partial_input:
        suggestions = st.session_state.chat_manager.autocomplete(partial_input, limit=5)
        if suggestions:
            st.write("**Suggestions:**")
            for s in suggestions:
                st.write(f"- {s}")
        else:
            st.write("No matches found...")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Here we want to provide a toggle or alternatively a button to enable RAG-mode
    # and thus make everything possible.
    rag_mode = st.sidebar.checkbox("Enable Rag Mode?")
    # Here we're going to provide a separate button to request a JSON output
    # that is structured. (You can combine them if you want, but for now let us
    # utilize a clean separation).
    generate_json_button = st.sidebar.button("Generate JSON Structure?")
    classify_button = st.sidebar.button("Classify Across Ontologies")
    # Normal chat input
    user_input = st.chat_input("What research topics are you interested in? Enter your query to classify:")
    if user_input:
        # And then we store this input in the state of the session so that we can later on reference it still
        st.session_state.last_user_input = user_input
        # Adds in the user message..adds it into the chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        # Thus we have to de-cide how we are going to do the query handling
        if rag_mode:
            # If we do find that we have enabled the RAG mode, then we can call the
            # generate_rag_response
            # response = st.session_state.chat_manager.generate_rag_response(user_input)
            # Instead of handle_message, we want to call the handle_rag_message
            response = st.session_state.chat_manager.handle_rag_message(user_input)
        else:
            # Other-wise, we're going to want to do in the flow for the normal
            # handle_message thing.
            response = st.session_state.chat_manager.handle_message(user_input)
        # Show the assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to message history
        st.session_state.messages.append({"role": "assistant", "content": response})
    # If the user presses the button for the "Generate JSON Structure?":
    if generate_json_button:
        if st.session_state.last_user_input:
            # Here we can call the method for structured generation in ChatManager
            structured_data = st.session_state.chat_manager.generate_structured_ontology(st.session_state.last_user_input)
            # And the JSON result is something that we directly display in the User Interface
            st.write("**Structured JSON Output:**")
            st.json(structured_data)
        else:
            st.warning("Please enter a query before generating JSON. No query text is available.")
    # Reset button
    if st.sidebar.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.chat_manager.reset_conversation()
        # Here we're going to additionally reset the last input of the user.
        st.session_state.last_user_input = ""
        st.experimental_rerun()
    if classify_button:
        user_topic = st.session_state.last_user_input
        if not user_topic:
            st.warning("No user query to classify. Please enter a query first.")
        else:
            alignment = st.session_state.chat_manager.handle_classification_message(user_topic)
            st.write("**Ontology Alignment Results:**")
            st.json(alignment)

    # In your main() or another function

    person_query = st.text_input("Search for a person by name or affiliation")
    if person_query:
        # search persons
        matched_persons = st.session_state.chat_manager.topic_agent.retrieve_persons(person_query)
        if matched_persons:
            st.write("Found the following persons:")
            for p in matched_persons:
                st.write(f"- {p['name']} ({p['affiliation']})")

            # Let user pick one to see JSON-LD
            selected_name = st.selectbox("Select a person to see JSON-LD", [p["name"] for p in matched_persons])
            if selected_name:
                # get that person's data
                person_data = next(x for x in matched_persons if x["name"] == selected_name)
                jsonld = st.session_state.chat_manager.topic_agent.generate_schema_org_person(person_data)
                st.write("**Schema.org JSON-LD:**")
                st.json(json.loads(jsonld))  # nicely formatted
        else:
            st.write("No persons found for that query.")

if __name__ == "__main__":
    main()
