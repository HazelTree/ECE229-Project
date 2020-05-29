import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
import analysis 

csv_path = '../data/bank-additional-full.csv'
my_analysis = analysis.Analysis(csv_path)
myList, labels = my_analysis.map_age()
print(1)
def marital_state_distribution():
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
