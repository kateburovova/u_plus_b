import pandas as pd
import streamlit as st
import plotly.express as px
import logging

from elasticsearch import Elasticsearch
logging.basicConfig(level=logging.INFO)

def get_unique_category_values(index_name, field, es_config):
    """
    Retrieve unique values from the field in a specified Elasticsearch index.
    Returns:
    list: A list of unique values from the 'category.keyword' field.
    """
    try:
        es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"], request_timeout=300)

        agg_query = {
            "size": 0,
            "aggs": {
                "unique_categories": {
                    "terms": {"field": field, "size": 10000}
                }
            }
        }

        response = es.search(index=index_name, body=agg_query)
        unique_values = [bucket['key'] for bucket in response['aggregations']['unique_categories']['buckets']]

        return unique_values
    except Exception as e:
        logging.error(f"Error retrieving unique values from {field}: {e}")
        return []

def populate_default_values(index_name, es_config):
    """
    Retrieves unique values for specified fields from an Elasticsearch index
    and appends an "Any" option to each list from the specified Elasticsearch index.
    """
    category_values = get_unique_category_values(index_name, 'category.keyword', es_config)
    language_values = get_unique_category_values(index_name, 'language.keyword', es_config)
    country_values = get_unique_category_values(index_name, 'country.keyword', es_config)
    category_values.append("Any")
    language_values.append("Any")
    country_values.append("Any")

    return sorted(category_values), sorted(language_values), sorted(country_values)

project_indexes = {
    'ua-by': [
        "ua-by-telegram",
        "ua-by-web",
        "ua-by-facebook",
        "ua-by-youtube"
    ]
}

flat_index_list = [index for indexes in project_indexes.values() for index in indexes]

def populate_terms(selected_items, field):
    """
    Creates a list of 'term' queries for Elasticsearch based on selected items.
    Returns:
        list: A list of 'term' queries for inclusion in an Elasticsearch 'should' clause.
    """
    if (selected_items is None) or ("Any" in selected_items):
        return []
    else:
        return [{"term": {field: item}} for item in selected_items]


def add_terms_condition(must_list, terms):
    """
    Adds individual term to Elasticsearch query.
    """
    if terms:
        must_list.append({
            "bool": {
                "should": terms,
                "minimum_should_match": 1
            }
        })


def create_must_term(category_terms, language_terms, country_terms, formatted_start_date, formatted_end_date):
    """
    Constructs a 'must' term for an Elasticsearch query that incorporates
    filters for date range, category, language, and country.

    Each filter term is added to the 'must' term only if it is not None.
    """

    must_term = [
        {"range": {"date": {"gte": formatted_start_date, "lte": formatted_end_date}}}
    ]

    add_terms_condition(must_term, category_terms)
    add_terms_condition(must_term, language_terms)
    add_terms_condition(must_term, country_terms)

    return must_term


def create_dataframe_from_response(response):
    """
    Creates a pandas DataFrame from Elasticsearch response data.
    Returns:
        pd.DataFrame: A DataFrame containing the selected fields from the response.
    """
    try:
        selected_documents = []

        if 'hits' not in response or 'hits' not in response['hits']:
            print("No data found in the response.")
            return pd.DataFrame()  # Return an empty DataFrame

        for doc in response['hits']['hits']:
            selected_doc = {
                'date': doc['_source'].get('date', ''),
                'text': doc['_source'].get('text', ''),
                # 'translated_text': doc['_source'].get('translated_text', ''),
                'url': doc['_source'].get('url', ''),
                'country': doc['_source'].get('country', 'None'),
                'language': doc['_source'].get('language', 'None'),
                'category': doc['_source'].get('category', 'None'),
                'id': doc.get('_id', '')
            }
            selected_documents.append(selected_doc)

        df_selected_fields = pd.DataFrame(selected_documents)

        if 'date' in df_selected_fields.columns:
            df_selected_fields['date'] = pd.to_datetime(df_selected_fields['date']).dt.date

        return df_selected_fields

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def display_distribution_charts(df):
    """
    Displays donut charts for category, language, and country distributions in Streamlit.
    The layout is three columns with one donut chart in each column.
    """

    if df.empty:
        st.write("No data available to display.")
        return

    col1, col2, col3 = st.columns(3)

    if 'category' in df.columns:
        category_counts = df['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        fig_category = px.pie(category_counts, names='category', values='count',
                              title='Category Distribution', hole=0.4)
        col1.plotly_chart(fig_category, use_container_width=True)

    if 'language' in df.columns:
        language_counts = df['language'].value_counts().reset_index()
        language_counts.columns = ['language', 'count']
        fig_language = px.pie(language_counts, names='language', values='count',
                              title='Language Distribution', hole=0.4)
        col2.plotly_chart(fig_language, use_container_width=True)

    if 'country' in df.columns:
        country_counts = df['country'].value_counts().reset_index()
        country_counts.columns = ['country', 'count']
        fig_country = px.pie(country_counts, names='country', values='count',
                             title='Country Distribution', hole=0.4)
        col3.plotly_chart(fig_country, use_container_width=True)

def create_dataframe_from_response_filtered(response, score_threshold=0.7):
    records = []
    for hit in response['hits']['hits']:
        if hit['_score'] >= score_threshold:
            source = hit['_source']
            similarity_score = hit['_score']
            source['similarity_score'] = similarity_score
            records.append(source)

    df = pd.DataFrame(records)

    return df

def search_elastic_below_threshold(es_config, selected_index, question_vector, must_term, max_doc_num=10000):
    try:
        es = Elasticsearch(f'https://{es_config["host"]}:{es_config["port"]}', api_key=es_config["api_key"],
                           request_timeout=600)

        response = es.search(index=selected_index,
                             size=max_doc_num,
                             knn={"field": "embeddings.WhereIsAI/UAE-Large-V1",
                                  "query_vector": question_vector,
                                  "k": 100,
                                  "num_candidates": 10000,
                                  # "similarity": 20, # l2 norm, so not the [0,1]
                                  "filter": {
                                      "bool": {
                                          "must": must_term
                                      }
                                  }
                                  }
                             )
        df = create_dataframe_from_response_filtered(response)
        return df

    except Exception as e:
        st.error(f'Failed to connect to Elasticsearch: {str(e)}')

        return None



