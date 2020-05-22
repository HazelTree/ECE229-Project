import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
from textwrap import dedent as d
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px


from visualization import analysis 

csv_path = 'data/bank-additional-full.csv'
my_analysis = analysis.Analysis(csv_path)
myList, labels = my_analysis.map_age()

def martial_state_distribution():
    '''
    This function gives the plot of distribution of people's martial status.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of martial status distribution.


    '''

    percents = my_analysis.percentage_of_population('marital')
    v = my_analysis.get_count('marital')['y']
    values = [v[1], v[0], v[2]]
    labels = ['Married', 'Divorced', 'Single']
    my_analysis.get_count('marital')
    explode = (0.2, 0, 0)
    fig = px.pie(percents,  values= values, names = labels, 
                title = 'age of people who are married, divorced and single')
    return fig



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Visualization', value='tab-1'),
        dcc.Tab(label='Prediction', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
        
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(children =[
            dcc.Graph(
                id = "martial status",
                figure = martial_state_distribution()
            )


        ] )
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])


if __name__ == '__main__':
    app.run_server(debug=True)