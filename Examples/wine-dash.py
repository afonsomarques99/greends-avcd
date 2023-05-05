# ATENTION: THIS CODE DOES NOT WORK IN JUPYTER NOTEBOOK! 
# Need to copy-paste to a *.py file and run in command line (see above).

# Import modules
# import Dash, dcc (stands for Dash Core Components - this module includes a Graph component called dcc.Graph, 
# which is used to render interactive graphs amd dcc.slider to render an interactive slider).
# We also import sklearn.decomposition.PCA to run a PCA, the plotly.express library to build the interactive graphs, 
# and pandas to work with DataFrames.

from dash import Dash, dcc, html, Input, Output
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv('winequality_red.csv')

# Initialize the app
# This line is known as the Dash constructor and is responsible for initializing your app. 
# It is almost always the same for any Dash app you create.
app = Dash(__name__)

# App layout
# The app layout represents the app components that will be displayed in the web browser, 
# normally contained within a html.Div.
app.layout = html.Div([
    html.H1("Visualization of PCA's explained variance", style={'textAlign':'center'}),
    dcc.Graph(id="pca-visualization-x-graph"),
    html.P("Number of components:"),
    dcc.Slider(
        id='pca-visualization-x-slider',
        min=2, max=11, value=2, step=1)
])

# Add controls to build the interaction
# The inputs and outputs of our app are the properties of a particular component. 
# The output is the figure property of the component with the ID "pca-visualization-x-graph"
# THe input is the value property of the component that has the ID "pca-visualization-x-slider".
# The callback function's argument 'n_components' refers to the component property of the input. 
# We build PCA plots inside the callback function, assigning the chosen value in the slider. 
# This means that every time the user selects the number of components with the slider, the figure is rebuilt
# to add more or less components
# Finally, we return the scatter plots at the end of the function. 
# This assigns the plots to the figure property of the dcc.Graph, thus displaying the figure in the app.
@app.callback(
    Output(component_id="pca-visualization-x-graph", component_property="figure"), 
    Input(component_id="pca-visualization-x-slider", component_property="value"))

def run_and_plot(n_components):
    df_wine = pd.read_csv('winequality_red.csv')
    df_wine2 = df_wine.iloc[:, 0:11]
    wine_scaled = StandardScaler().fit_transform(df_wine2)
    df = pd.DataFrame(data=wine_scaled, columns=df_wine2.columns)
    pca = PCA(n_components=n_components) # defines the number of components in the PCA
    components = pca.fit_transform(df) # fits a PCA
    var = pca.explained_variance_ratio_.sum() * 100 # % of explained variance by each PC
    labels = {str(i): f"PC {i+1}" for i in range(n_components)} # PC labels
    labels['color'] = 'quality'
    fig = px.scatter_matrix(
        components,
        color=df_wine['quality'],
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {var:.2f}%',
        width=1400, height=1300
        )
    fig.update_traces(diagonal_visible=False)
    return fig

# Run the app - These lines are for running your app, and they are almost always the same for any Dash app you create.
if __name__ == "__main__":
    app.run_server(debug=True)