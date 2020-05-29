import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from textwrap import dedent as d
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.offline import plot
from sklearn.externals import joblib
from util import dynamic_predict


from visualization import analysis  
# from visualization import plots 

csv_path = 'data/bank-additional-full.csv'
my_analysis = analysis.Analysis(csv_path)
myList, labels = my_analysis.map_age()

# Read and modify prediction data
predictions = pd.read_csv('data/predictions.csv')

df = predictions.copy()

df = df[['customer_id', 'age', 'job_transformed', 'poutcome', 'pred', 'prob_1']]  # prune columns for example
df.sort_values(by = ['prob_1'], ascending = False, inplace = True)
df['prob_1'] = np.around(df['prob_1'], decimals = 2)
df['is_called'] = 0

def marital_state_distribution():
    '''
    This function gives the plot of distribution of people's marital status.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.


    '''
    percents = my_analysis.percentage_of_population('marital')
    v = my_analysis.get_count('marital')['y']
    values = [v[1], v[0], v[2]]
    labels = ['Married', 'Divorced', 'Single']
    my_analysis.get_count('marital')
    explode = (0.2, 0, 0)
    fig = px.pie(percents,  values= values, names = labels, 
                title = 'Number of people who are married, divorced and single')
    return fig

def marital_status_probab():
    marital_status_probab = my_analysis.get_probabilities('marital')
    data = marital_status_probab
    data['y'] = data['y']*100
    fig = px.bar(data, x='marital', y='y',
                hover_data=data, color='marital',labels={'y':'Probability of Success (%)', 'marital': 'Marital Status'},
                height=400)
    return fig

def education_level_distribution():
    edu_count = my_analysis.get_count('education')
    edu_success_count = my_analysis.get_success_count('education')

    status=['Dropout', 'illiterate', 'professional.course', 'university.degree']

    fig = go.Figure(data=[
        go.Bar(name='Not Successful', x=status, y=edu_count['y']-edu_success_count['y']),
        go.Bar(name='Success', x=status, y=edu_success_count['y'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack', xaxis_title="Education Level", yaxis_title="Number of people")
    return fig

def education_level_prob():
    edu_prob = my_analysis.get_probabilities('education')
    data = edu_prob
    data['y'] = data['y']*100
    fig = px.bar(data, x='education', y='y',
                hover_data=data, color='education',labels={'y':'Probability of Success (%)', 'education': 'Education Level'},
                height=400)
    return fig

def income_level_distribution():
    job_count = my_analysis.get_count('job')
    job_success_count = my_analysis.get_success_count('job')
    #print(job_count)
    status=['higher income', 'lower income', 'no income']

    fig = go.Figure(data=[
        go.Bar(name='Not Successful', x=status, y=job_count['y']-job_success_count['y']),
        go.Bar(name='Success', x=status, y=job_success_count['y'])
    ])
    fig.update_layout(barmode='stack', xaxis_title="Income Level", yaxis_title="Number of people")
    return fig

def job_prob():
    job_prob = my_analysis.get_probabilities('job')
    data = job_prob
    data['y'] = data['y']*100
    fig = px.bar(data, x='job', y='y',
                hover_data=data, color='job',labels={'y':'Probability of Success (%)', 'job': 'Job'},
                height=400)
    return fig

def contact_way_distribution():
    contact_count = my_analysis.get_count('contact')
    contact_success_count = my_analysis.get_success_count('contact')
    status=['cellular', 'telephone']

    fig = go.Figure(data=[
        go.Bar(name='Not Successful', x=status, y=contact_count['y']-contact_success_count['y']),
        go.Bar(name='Success', x=status, y=contact_success_count['y'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack', xaxis_title="Contact type", yaxis_title="Number of people")
    return fig

def contact_prob():
    contact_prob = my_analysis.get_probabilities('contact')
    data = contact_prob
    data['y'] = data['y']*100
    fig = px.bar(data, x='contact', y='y',
                hover_data=data, color='contact',labels={'y':'Probability of Success (%)', 'contact': 'Contact type'},
                height=400)
    return fig

def loan_status():
    loan_count = my_analysis.get_count('loan')
    loan_success_count = my_analysis.get_success_count('loan')
    status=['yes', 'no', 'Info Not Available']

    fig = go.Figure(data=[
        go.Bar(name='Not Successful', x=status, y=loan_count['y']-loan_success_count['y']),
        go.Bar(name='Success', x=status, y=loan_success_count['y'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack', xaxis_title="Do people have a loan?", yaxis_title="Number of people")
    return fig

def loan_prob():
    loan_prob = my_analysis.get_probabilities('loan')
    data = loan_prob
    data['y'] = data['y']*100
    fig = px.bar(data, x='loan', y='y',
                hover_data=data, color='loan',labels={'y':'Probability of Success (%)', 'loan': 'Do people have a loan?'},
                height=400)
    return fig

def house_status_distribution():
    housing_count = my_analysis.get_count('housing')
    housing_success_count = my_analysis.get_success_count('housing')
    status=['yes', 'no', 'Info Not Available']

    fig = go.Figure(data=[
        go.Bar(name='Not Successful', x=status, y=housing_count['y']-housing_success_count['y']),
        go.Bar(name='Success', x=status, y=housing_success_count['y'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack', xaxis_title="Housing Status", yaxis_title="Number of people")
    return fig

def house_prob():
    housing_prob = my_analysis.get_probabilities('housing')
    data = housing_prob
    data['y'] = data['y']*100
    fig = px.bar(data, x='housing', y='y',
                hover_data=data, color='housing',labels={'y':'Probability of Success (%)', 'housing': 'Housig Status'},
                height=400)
    return fig

def prediction_pie_chart():
    '''
    Plot predicted telecaller success ratios on the test data.
    
    :return: Plots the prediction pie chart.
    :rtype: plotly.graph_objs._figure.Figure
    '''
    fig = px.pie(predictions, 
                 values=[1 for i in range(len(predictions))], 
                 names= np.where(predictions['pred'] == 1, 'Purchase', 'No Purchase'), 
                 title='Overall Telemarketing Success Predictions')
    return fig

def predicted_prob_hist():
    '''
    Plot the histogram of the predicted probabilities.
    
    :return: Plots the prediction probability histogram.
    :rtype: plotly.graph_objs._figure.Figure
    '''
    fig = px.histogram(predictions, 
                       x="prob_1", 
                       nbins=5, 
                       labels = {'prob_1' : 'Success Probabilty'},
                       title='Histogram of Success Probabilities')
    return fig

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

tab_style = {
    'fontWeight': 'bold'
}
vis_tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '15px',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '15px'
}
app.layout = html.Div(children = [
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Visualization', value='tab-1', style=tab_style),
        dcc.Tab(label='Prediction', value='tab-2', style=tab_style),
        dcc.Tab(label='Prediction-test', value='tab-new', style=tab_style),
    ]),
    html.Div(id='tabs-content')
])
layout_tab_1  = html.Div(children = [
    dcc.Tabs(id = "vis-tabs", value = "vistab", vertical=True, parent_style={'float': 'left','width': '40'},children =[
        dcc.Tab(label='Marital Status', value='tab-3', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Educational Level', value='tab-4', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Income&Job', value='tab-5', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Contact Type', value='tab-6', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Loan Status', value='tab-7', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Housing Status', value='tab-8', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Prediction Overview', value='tab-9', style=vis_tab_style, selected_style=tab_selected_style),
    ]),
    html.Div(id='vis-tabs-content',style={'float': 'right'})
])

marital_status_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = marital_state_distribution()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center' }),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = marital_status_probab()
            )],
            style={'height': 400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

        ])
educational_Level_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = education_level_distribution()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = education_level_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])

income_vis = html.Div(children =[

            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = income_level_distribution()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = job_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])

contact_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = contact_way_distribution()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = contact_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])

loan_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = loan_status()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = loan_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])
house_vis = html.Div(children =[            
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = house_status_distribution()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = house_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])


prediction_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "prediction_pie_chart",
                figure = prediction_pie_chart()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "predicted_prob_hist",
                figure = predicted_prob_hist()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

        ])

@app.callback(Output('vis-tabs-content', 'children'),
              [Input('vis-tabs', 'value')])
def render_content(tab):
    if tab == 'tab-3':
        return marital_status_vis
    elif tab == 'tab-4':
        return educational_Level_vis
    elif tab == 'tab-5':
        return income_vis 
    elif tab == 'tab-6':
        return contact_vis 
    elif tab == 'tab-7':
        return loan_vis 
    elif tab == 'tab-8':
        return house_vis 
    elif tab == 'tab-9':
        return prediction_vis  
    else:
        return marital_status_vis


layout_tab_2 = html.Div(children =[
             
             html.Div(dash_table.DataTable(
                         columns=[
                             {'name': 'Customer ID', 'id': 'customer_id', 'type': 'numeric', 'editable': False},
                             {'name': 'Age', 'id': 'age', 'type': 'numeric', 'editable': False},
                             {'name': 'Income', 'id': 'job_transformed', 'type': 'text', 'editable': False},
                             {'name': 'Previously Contacted', 'id': 'poutcome', 'type': 'text', 'editable': False},
                             {'name': 'Prediction', 'id': 'pred', 'type': 'numeric', 'editable': False},
                             {'name': 'Probability of Success', 'id': 'prob_1', 'type': 'numeric', 'editable': False},
                             {'name': 'Is Called', 'id': 'is_called', 'type': 'numeric', 'editable': True}
                         ],
                         data=df.to_dict('records'),
                         filter_action='native',
                    
                         style_table={
                             'height': 400,
                         },
                         style_data={
                             'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
                             'overflow': 'hidden',
                             'textOverflow': 'ellipsis',
                         }
                         )                    
                     )                
                
        ])

layout_tab_new = html.Div(children =[
    html.Div(children =[
    html.Label('Enter years of experience: '),
    dcc.Input(id='nremployed', placeholder='nr.employed', type='text'),
    html.Label('Enter years of experience: '),
    dcc.Input(id='poutcome_success', placeholder='poutcome_success', type='number', min=0, max=1, step=1),
    html.Label('Enter years of experience: '),
    dcc.Input(id='emp', placeholder='emp.var.rate', type='text'),
    ],style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),

    html.Div(children =[
    html.Label('Enter years of experience: '),
    dcc.Input(id='pdays', placeholder='pdays', type='text'),
    html.Label('Enter years of experience: '),
    dcc.Input(id='consconfidx', placeholder='cons.conf.idx', type='text'),
    html.Label('Enter years of experience: '),
    dcc.Input(id='euribor3m', placeholder='euribor3m', type='text'),
    html.Label('Enter years of experience: '),
    dcc.Input(id='job_transformed_no_income', placeholder='job_transformed_no_income', type='number', min=0, max=1, step=1),
    ],style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),


    html.Div(children=[
        html.H1(children='Probability of Success: '),
        html.Div(id='pred-output')
    ], style={'textAlign': 'center'}),
])
@app.callback(
    Output('pred-output', 'children'),
    [Input('nremployed', 'value'),
     Input('poutcome_success', 'value'),
     Input('emp', 'value'),
     Input('pdays', 'value'),
     Input('consconfidx', 'value'),
     Input('euribor3m', 'value'),
     Input('job_transformed_no_income', 'value')])
def show_success_probability(nr_employed, poutcome_success, emp_var_rate, pdays, cons_conf, euribor, no_income):
    if not nr_employed or not poutcome_success or not emp_var_rate or not pdays or not cons_conf or not euribor or not no_income:
        raise PreventUpdate
    else:
        prob = round(dynamic_predict(float(nr_employed), float(poutcome_success), float(emp_var_rate), float(pdays), float(cons_conf), float(euribor), float(no_income))[0],4)*100
        return html.Div(children =[
            html.H1(children=str(prob)+"%")
        ])


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return layout_tab_1
    elif tab == 'tab-2':
        return layout_tab_2
    elif tab == "tab-new":
        return layout_tab_new

if __name__ == '__main__':
    model = joblib.load("LR_prediction.joblib")
    app.run_server(debug=True)
