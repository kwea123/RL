import dash
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import socket
import threading

X = deque(maxlen=20)
X.append(0)
Y = deque(maxlen=20)
Y.append(0)
N = 1

print("Listening")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '0.0.0.0'
port = 5555
sock.bind((host, port))
sock.listen()
(client_sock, client_addr) = sock.accept()
print("Client Info: ", client_sock, client_addr)

def get_cmsg():
    global X, Y, N
    while True:
        msg = client_sock.recv(4).decode()
        X.append(random.random())
        Y.append(random.random())
        N += 1
        # X.append(X[-1]+1)
        # Y.append(float(msg))
    
threading.Thread(target=get_cmsg, args=(), name='msg_thread').start()

app = dash.Dash()
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=False),
        dcc.Interval(
            id='graph-update',
            interval=1000
        ),
    ]
)

@app.callback(Output('live-graph', 'figure'),
              events=[Event('graph-update', 'interval')])
def update_graph_scatter():
    # data = go.Scatter(
    #         x=list(X),
    #         y=list(Y),
    #         name='Scatter',
    #         mode='lines+markers'
    #         )
    # return {'data': [data],'layout' : go.Layout(title=str(X[-1])+" "+str(Y[-1]),
    #                                             xaxis={'range':[min(X),max(X)]},
    #                                             yaxis={'range':[min(Y),max(Y)]},)}
    start = 1
    if N >= 20:
        start = N-19
    data = [go.Scatter(
        x = list(X),
        y = list(Y),
        text = list(range(start,N+1)),
        textposition='top right',
        textfont=dict(
            family='sans serif',
            size=9,
            color='red'
        ),
        mode='lines+markers+text',
        line=dict(
            width=2,
        ),
        marker=dict(
            size=10,
            color = list(range(start,N+1)),
            colorscale='Viridis',
            showscale=True,
        )
    )]

    layout = dict(
    plot_bgcolor ='white',
    width = 1000,
    xaxis= {'range':[-1.1,1.1], 'zeroline': False},
    yaxis= {'range':[-1.1,1.1], 'zeroline': False},
    )
    
    return dict(data=data, layout=layout)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)