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

print("Listening")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '127.0.0.1'
port = 5555
sock.bind((host, port))
sock.listen()
(client_sock, client_addr) = sock.accept()
print("Client Info: ", client_sock, client_addr)

def get_cmsg():
    global X, Y
    while True:
        msg = client_sock.recv(4).decode()
        X.append(X[-1]+1)
        Y.append(float(msg))
    
threading.Thread(target=get_cmsg, args=(), name='msg_thread').start()

app = dash.Dash()
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=100
        ),
    ]
)

@app.callback(Output('live-graph', 'figure'),
              events=[Event('graph-update', 'interval')])
def update_graph_scatter():
    data = go.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode='lines+markers'
            )
    return {'data': [data],'layout' : go.Layout(title=str(X[-1])+" "+str(Y[-1]),
                                                xaxis={'range':[min(X),max(X)]},
                                                yaxis={'range':[min(Y),max(Y)]},)}

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)