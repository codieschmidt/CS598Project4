import pandas as pd
import requests
import json
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from dash.dependencies import ALL, State

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

cosine_similarity_matrix = json.loads(requests.get('https://raw.githubusercontent.com/codieschmidt/CS598Project4/main/cosine_similarity_matrix.json').content)

movie_list_index = np.argsort(movies['movie_id'].unique())
movie_list = np.take_along_axis(movies['movie_id'].unique(), movie_list_index, axis=0)

def get_displayed_movies():
    return movies.head(100)

def get_movie_name_from_index(index, movies):
    
    return movies['title'][index]

def get_recommended_movies(new_user_ratings):

    newUser = np.zeros(3706)
    
    for key in new_user_ratings:
        newUser[key] = new_user_ratings[key]
    
    predicted_ratings = []
    
    predicted_ratings = []
    
    print('Generating predicitons')
    for m in range(len(newUser)):
        #newUser[m] == 0 prevents pciking movies in the prediction vector
        if sum(cosine_similarity_matrix[m]) != 0 and newUser[m] == 0:
            prediction = np.dot(cosine_similarity_matrix[m],newUser)*(1/(sum(cosine_similarity_matrix[m])))
            predicted_ratings.append([prediction, movie_list[m]])       
        else:
            predicted_ratings.append([0, movie_list[m]])       
                        
    print('Sorting predicitons')
    sorted_prediciton_ratings = np.array(predicted_ratings) 
    sorted_prediciton_ratings_idx = np.flip(np.argsort(sorted_prediciton_ratings[:,0], axis=0))
    #print(sorted_prediciton_ratings)
    sorted_prediciton_ratings_idx = sorted_prediciton_ratings_idx[:11]
                      
    print('Getting top 10')
    top_10_movie_id = []
    top_10_movie_title = []
    #get top 10 movies
    for i in range(10):            
        #print(sorted_prediciton_ratings_idx[i])
        top_10_movie_id.append(sorted_prediciton_ratings_idx[i])
        top_10_movie_title.append(movies[movies['movie_id'] == sorted_prediciton_ratings_idx[i]]['title'].values[0])
    
    d = {'movie_id': top_10_movie_id, 'title': top_10_movie_title}
    df = pd.DataFrame(data=d)
                             
                                
    return df

def get_popular_movies(genre: str):
    
    recs = json.loads(requests.get('https://raw.githubusercontent.com/codieschmidt/CS598Project4/main/genre_recommendations.json').content)
    recs = pd.DataFrame.from_dict(recs[genre]).rename(columns={"MovieID": "movie_id", "Title": "title"})
    recs = pd.DataFrame(recs, columns=['movie_id', 'title', 'rating'])
    
    return recs
    
#from myfuns import (genres, get_displayed_movies, get_popular_movies, get_recommended_movies)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], 
               suppress_callback_exceptions=True)
server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H3("Movie Recommender", className="display-8"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("System 1 - Genre", href="/", active="exact"),
                dbc.NavLink("System 2 - Collaborative", href="/system-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])

def render_page_content(pathname):
    if pathname == "/":
        return html.Div(
            [
                html.H1("Select a genre"),
                dcc.Dropdown(
                    id="genre-dropdown",
                    options=[{"label": k, "value": k} for k in genres],
                    value=None,
                    className="mb-4",
                ),
                html.Div(id="genre-output", className=""),
            ]
        )
    elif pathname == "/system-2":
        movies = get_displayed_movies()
        return html.Div(
            [
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H1("Rate some movies below to"),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        children=[
                                            "Get recommendations ",
                                            html.I(className="bi bi-emoji-heart-eyes-fill"),
                                        ],
                                        size="lg",
                                        className="btn-success",
                                        id="button-recommend",
                                    ),
                                    className="p-0",
                                ),
                            ],
                            className="sticky-top bg-white py-2",
                        ),
                        html.Div(
                            [
                                get_movie_card(movie, with_rating=True)
                                for idx, movie in movies.iterrows()
                            ],
                            className="row row-cols-1 row-cols-5",
                            id="rating-movies",
                        ),
                    ],
                    id="rate-movie-container",
                ),
                html.H1(
                    "Your recommendations", id="your-recommendation",  style={"display": "none"}
                ),
                dcc.Loading(
                    [
                        dcc.Link(
                            "Try again", href="/system-2", refresh=True, className="mb-2 d-block"
                        ),
                        html.Div(
                            className="row row-cols-1 row-cols-5",
                            id="recommended-movies",
                        ),
                    ],
                    type="circle",
                ),
            ]
        )

@app.callback(Output("genre-output", "children"), Input("genre-dropdown", "value"))
def update_output(genre):
    if genre is None:
        return html.Div()
    else: 
        return [
            dbc.Row(
                [
                    html.Div(
                        [
                            *[
                                get_movie_card(movie)
                                for idx, movie in get_popular_movies(genre).iterrows()
                            ],
                        ],
                        className="row row-cols-1 row-cols-5",
                    ),
                ]
            ),
        ]


    
def get_movie_card(movie, with_rating=False):
    return html.Div(
        dbc.Card(
            [
                dbc.CardImg(
                    src=f"https://liangfgithub.github.io/MovieImages/{movie.movie_id}.jpg?raw=true",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        html.H6(movie.title, className="card-title text-center"),
                    ]
                ),
            ]
            + (
                [
                    dcc.RadioItems(
                        options=[
                            {"label": "1", "value": "1"},
                            {"label": "2", "value": "2"},
                            {"label": "3", "value": "3"},
                            {"label": "4", "value": "4"},
                            {"label": "5", "value": "5"},
                        ],
                        className="text-center",
                        id={"type": "movie_rating", "movie_id": movie.movie_id},
                        inputClassName="m-1",
                        labelClassName="px-1",
                    )
                ]
                if with_rating
                else []
            ),
            className="h-100",
        ),
        className="col mb-4",
    )
    
@app.callback(
    Output("rate-movie-container", "style"),
    Output("your-recommendation", "style"),
    [Input("button-recommend", "n_clicks")],
    prevent_initial_call=True,
)    
def on_recommend_button_clicked(n):
    return {"display": "none"}, {"display": "block"}

@app.callback(
    Output("recommended-movies", "children"),
    [Input("rate-movie-container", "style")],
    [
        State({"type": "movie_rating", "movie_id": ALL}, "value"),
        State({"type": "movie_rating", "movie_id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)

def on_getting_recommendations(style, ratings, ids):
    rating_input = {
        ids[i]["movie_id"]: int(rating) for i, rating in enumerate(ratings) if rating is not None
    }
  
    recommended_movies = get_recommended_movies(rating_input)
 
    return [get_movie_card(movie) for idx, movie in recommended_movies.iterrows()]


@app.callback(
    Output("button-recommend", "disabled"),
    Input({"type": "movie_rating", "movie_id": ALL}, "value"),
)
def update_button_recommened_visibility(values):
    return not list(filter(None, values))

if __name__ == "__main__":
    app.run_server(debug=False)
