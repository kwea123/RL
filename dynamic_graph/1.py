import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html


#fig1 = {'data':[{'x':[1,2,3,4,5], 'y':[5,6,1,8,6], 'type':'line', 'name':'boats'},
#                {'x':[1,2,3,4,5], 'y':[1,4,1,6,3], 'type':'bar', 'name':'cars'}],
#        'layout':{'title':'basic'}
#       }

app = dash.Dash()
app.layout = html.Div(children=[dcc.Input(id='input', value='', type='text'), html.Div(id='output')])

@app.callback(
    Output(component_id='output', component_property='children'),
    [Input(component_id='input', component_property='value')]
)
def update_value(input_data):
    return 'Input: "{}"'.format(input_data)

if __name__ == '__main__':
    app.run_server(debug=True)