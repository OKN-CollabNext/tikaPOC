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

def main():
    """ Here is where we grab any existing parameters for the query.. """
    params = st.query_params
    topic_id = params.get("topic_id", [None])[0]
    """ And now this is the important step..if we do have a topic_id, then
    we need to show the "Topic Details" page.  """
    if topic_id is not None:
        st.title(f"Topic Details for {topic_id}")
        """ and here's the example..we might be able to query the DataBase for the full topic info..here we'll pretend there's a get_topic_info() method below..and that is what returns the { 'id': ..., 'display_name': ..., 'description': ... }
        And when we do, we will know the topic info.  """
        topic_info = get_topic_info_from_db(topic_id)
        if topic_info:
            st.subheader(topic_info["display_name"])
            st.write(topic_info["description"])
            st.write("Number of matching keywords:", topic_info.get("matching_keywords", 0))
            """ And so on and so forth.  """
        """ Now, it is time to add in a link and or a button which will allow us to return to chat mode..which means we have got to clear the param.. """
        if st.button("Go Back"):
            st.set_query_params()
            """ and then that's what we do with the parametric clearance. """
            st.rerun()
        return
        """ And here we know that we are finished displaying the view in all its detailed glory """
    initialize_session_state()
    """ And if we don't find the topic_id, we do the usual chat """
    st.title("Socratic R.A.G. Agent")
    # Is it or is it not ready to load more? if it is then we can insert the button
    # below or maybe in the sidebar, with regard to the chat!
    load_more_button_pressed = st.button("Load More Results")
    """ Then we can allow the user(s) to put this message """
    if prompt := st.chat_input("What research topics are you interested in?", key="first_chat_input"):
        """ What's the user prompt..and what's the boolean for loading more..
        send those. """
        response = st.session_state.chat_manager.handle_message(
            user_message=prompt,
            load_more_button_pressed=load_more_button_pressed
        )
        st.write(response)
    """ The idea is to display the chat messages """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    #Input to the chat
    if prompt := st.chat_input("What research topics are you interested in?", key="second_chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.session_state.chat_manager.handle_message(prompt)
            st.markdown(response, unsafe_allow_html=True)
            """ We do allow the HTML for the links """
        st.session_state.messages.append({"role": "assistant", "content": response})
    """ And add on the re-set button onto the sidebar """
    with st.sidebar:
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chat_manager.reset_conversation()
            st.set_query_params()
            st.rerun()

def get_topic_info_from_db(topic_id: str):
    """ And this is just our function placeholder. In practice, I'd call the existing
    database code and or make this new type of method in the ChatManager or alternately
    the TopicAgent that queries the table 'topics' by ID.
    For example:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, display_name, description FROM topics WHERE id = %s", (topic_id,))
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "display_name": row[1],
                    "description": row[2]
                }
    return None
    """
    return {
        "id": topic_id,
        "display_name": f"Topic {topic_id}",
        "description": "This is a dummy description for demonstration."
    }

if __name__ == "__main__":
    main()
