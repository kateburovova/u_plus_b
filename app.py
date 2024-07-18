import os
import streamlit as st
import time

import plotly.express as px
import streamlit.components.v1 as components

from langchain import hub
from datetime import datetime
from langchain import callbacks
from elasticsearch import Elasticsearch
from elasticsearch import BadRequestError
from elasticsearch.exceptions import NotFoundError
from angle_emb import AnglE, Prompts
from langchain_openai import ChatOpenAI
from authentificate import check_password
from utils import (display_distribution_charts,populate_default_values, project_indexes,
                   populate_terms,create_must_term, create_dataframe_from_response,flat_index_list,
                   search_elastic_below_threshold)


# Init Langchain and Langsmith services
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"rag_app : summarization : production : uaby_client"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets['ld_rag']['LANGCHAIN_API_KEY']
os.environ["LANGSMITH_ACC"] = st.secrets['ld_rag']['LANGSMITH_ACC']


# Init openai model
OPENAI_API_KEY = st.secrets['ld_rag']['OPENAI_KEY_ORG']
llm_chat = ChatOpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY,
             model_name='gpt-4-1106-preview')

es_config = {
    'host': st.secrets['ld_rag']['ELASTIC_HOST'],
    'port': st.secrets['ld_rag']['ELASTIC_PORT'],
    'api_key': st.secrets['ld_rag']['ELASTIC_API']
}

########## APP start ###########
st.set_page_config(layout="wide")

logo_url = 'assets/Blue_black_long.png'
st.image(logo_url,  width=300)

# description
st.markdown('App relies on data, collected and enriched by our team and provides citations for all sources used for '
            'answers. \n'
            'If you are running this app from a mobile device, tap on any '
            'empty space to apply changes to input fields. '
            'If you experience any technical issues, please [submit the form](https://docs.google.com/forms/d/e/1FAIpQLSfZTr4YoXXsjOOIAMVGYCeGgXd6LOsCQusctJ7hZODaW5HzGQ/viewform?pli=1) by selecting "LD app Technical Issue" for '
            'the type of request. To give feedback for request output, '
            'please use the feedback form at the end of the page.')

with open('assets/How_to.md', 'r') as file:
    markdown_content = file.read()

with st.expander("Learn more about the app"):
    st.markdown(markdown_content, unsafe_allow_html=True)

# Authorise user
if not check_password():
    st.stop()

# Get input parameters
st.markdown('### Please select search parameters ')

url = f'{os.environ["LANGSMITH_ACC"]}/simple-rag:9388b291'
prompt_template = hub.pull(url)

# Mappings for indexes
index_display_mapping = {
    "telegram": "ua-by-telegram",
    "web": "ua-by-web",
    "facebook": "ua-by-facebook",
    "youtube": "ua-by-youtube"
}

selected_indexes = []

# Reverse mapping to display names
display_index_mapping = {v: k for k, v in index_display_mapping.items()}

# Mapping for project
project_display_mapping = {
    "Ua By": "ua-by"
}

# Reverse mapping for projects
display_project_mapping = {v: k for k, v in project_display_mapping.items()}

selected_index = None
search_option = st.radio("Choose 'Specific platforms' if you want to search one or more different platforms, choose 'All platforms for project' to select all platforms within a project.",
                         ['Specific platforms', 'All platforms for project'])

category_terms = None
language_terms = None
country_terms = None

if search_option == 'Specific platforms':
    display_indexes = [display_index_mapping[index] for index in flat_index_list]
    selected_display_indexes = st.multiselect('Please choose one or more platforms', display_indexes, default=None, placeholder="Select one or more platforms")
    if selected_display_indexes:
        selected_indexes = [index_display_mapping[display_index] for display_index in selected_display_indexes]
        st.write(f"We'll search in: {', '.join(selected_display_indexes)}")
    else:
        selected_indexes = []
    selected_index = ",".join(selected_indexes)
else:
    display_projects = [display_project_mapping[project] for project in project_indexes.keys()]
    project_choice_display = st.selectbox('Please choose a project', display_projects, index=None, placeholder="Select project")
    if project_choice_display:
        project_choice = project_display_mapping[project_choice_display]
        selected_indexes = project_indexes.get(project_choice, [])
        st.write(f"We'll search in: {', '.join([display_index_mapping[idx] for idx in selected_indexes])}")
    selected_index = ",".join(selected_indexes)

if selected_index:
    category_values, language_values, country_values = populate_default_values(selected_index, es_config)

    with st.popover("Tap to refine filters"):
        st.markdown("Hihi 👋")
        st.markdown("If Any remains selected or no values at all, filtering will not be applied to this field. Start typing to find the option faster.")
        categories_selected = st.multiselect('Select "Any" or choose one or more categories', category_values, default=['Any'])
        languages_selected = st.multiselect('Select "Any" or choose one or more languages', language_values, default=['Any'])
        countries_selected = st.multiselect('Select "Any" or choose one or more countries', country_values, default=['Any'])

    category_terms = populate_terms(categories_selected, 'category.keyword')
    language_terms = populate_terms(languages_selected, 'language.keyword')
    country_terms = populate_terms(countries_selected, 'country.keyword')


    # Get input dates
    default_start_date = datetime(2024, 1, 1)
    default_end_date = datetime(2024, 1, 15)

    selected_start_date = st.date_input("Select start date:", default_start_date)
    formatted_start_date = selected_start_date.strftime("%Y-%m-%d")
    st.write("You selected start date:", selected_start_date)
    selected_end_date = st.date_input("Select end date:", default_end_date)
    formatted_end_date = selected_end_date.strftime("%Y-%m-%d")
    st.write("You selected end date:", selected_end_date)
    must_term = create_must_term(category_terms,
                                 language_terms,
                                 country_terms,
                                 formatted_start_date=formatted_start_date,
                                 formatted_end_date=formatted_end_date)

# Create prompt vector
input_question = None
st.markdown('### Please enter your question')
input_question = st.text_input("Enter your question here (phrased as if you ask a human)")


if input_question:

    formatted_start_date, formatted_end_date = None, None
    @st.cache_resource(hash_funcs={"_thread.RLock": lambda _: None, "builtins.weakref": lambda _: None})
    def load_model():
        angle_model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1',
                                            pooling_strategy='cls')
        return angle_model

    # Create question embedding
    angle = load_model()
    vec = angle.encode({'text': input_question}, to_numpy=True, prompt=Prompts.C)
    question_vector = vec.tolist()[0]

    # # Get input dates
    # default_start_date = datetime(2024, 1, 1)
    # default_end_date = datetime(2024, 1, 15)
    #
    # selected_start_date = st.date_input("Select start date:", default_start_date)
    # formatted_start_date = selected_start_date.strftime("%Y-%m-%d")
    # st.write("You selected start date:", selected_start_date)
    # selected_end_date = st.date_input("Select end date:", default_end_date)
    # formatted_end_date = selected_end_date.strftime("%Y-%m-%d")
    # st.write("You selected end date:", selected_end_date)
    # must_term = create_must_term(category_terms,
    #                              language_terms,
    #                              country_terms,
    #                              formatted_start_date=formatted_start_date,
    #                              formatted_end_date=formatted_end_date)


    if formatted_start_date and formatted_end_date:

        # Run search
        if st.button('RUN SEARCH', type="primary"):
            start_time = time.time()
            max_doc_num=30
            try:
                texts_list = []
                st.write(f'Running search for relevant posts for question: {input_question}')
                try:
                    es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"], request_timeout=600)
                except Exception as e:
                    st.error(f'Failed to connect to Elasticsearch: {str(e)}')

                response = es.search(index=selected_index,
                                     size=max_doc_num,
                                     knn={"field": "embeddings.WhereIsAI/UAE-Large-V1",
                                          "query_vector":  question_vector,
                                          "k": max_doc_num,
                                          "num_candidates": 10000,
                                          "filter": {
                                              "bool": {
                                                  "must": must_term,
                                                  "must_not": [{"term": {"type": "comment"}}]
                                              }
                                          }
                                          }
                                     )

                for doc in response['hits']['hits']:
                    texts_list.append((doc['_source']['translated_text'], doc['_source']['url']))

                st.write("Searching for documents, please wait 15 seconds on average to finish...")

                # Format urls so they work properly within streamlit
                corrected_texts_list = [(text, 'https://' + url if not url.startswith('http://') and not url.startswith(
                    'https://') else url) for text, url in texts_list]

                # Get summary for the retrieved data
                customer_messages = prompt_template.format_messages(
                    question=input_question,
                    texts=corrected_texts_list)

                # Print GPT summary
                st.markdown(f'### This is a summary for your question:')

                with callbacks.collect_runs() as cb:
                    st.write_stream(llm_chat.stream(customer_messages))
                    run_id = cb.traced_runs[0].id

                # st.markdown(content)
                st.write('******************')
                end_time = time.time()

                # Display tables
                st.markdown(f'### These are top {max_doc_num} texts used for summary generation:')
                df = create_dataframe_from_response(response)
                st.dataframe(df)
                display_distribution_charts(df)

                # Send rating to Tally
                execution_time = round(end_time - start_time, 2)
                tally_form_url = f'https://tally.so/embed/n0PA7P?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1&run_id={run_id}&time={execution_time}'
                components.iframe(tally_form_url, width=700, height=800, scrolling=True)


            except BadRequestError as e:
                st.error(f'Failed to execute search (embeddings might be missing for this index): {e.info}')
            except NotFoundError as e:
                st.error(f'Index not found: {e.info}')
            except Exception as e:
                st.error(f'An unknown error occurred: {str(e)}')

        if st.button('RE-RUN APP'):
            time.sleep(1)