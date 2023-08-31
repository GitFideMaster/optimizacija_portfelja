import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

import traceback

import os
import time

from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib
matplotlib.use('Agg')

import riskfolio as rp
import plotly.express as px

from io import BytesIO
import base64
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from dohvacanje_zg_burza_podataka import *

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

from dash import callback_context

custom_card_style = {
    'height': '100%',
    'display': 'flex',
    'flex-direction': 'column',
    'justify-content': 'space-between',
}

# Updated app layout
app.layout = dbc.Container(
    [
        dcc.Interval(
            id="dnevno_dohvacanje_podataka",
            interval=24 * 60 * 60 * 1000,  # 24 hours in milliseconds
            n_intervals=0
        ),

        dbc.Row(
            [
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Label("Odabir metrike optimizacije:", className="form-label"),
                                                    width=5
                                                ),
                                                dbc.Col(
                                                    dcc.Dropdown(
                                                        id='metric-dropdown',
                                                        options=[{'label': i, 'value': i} for i in ['Sharpeov omjer', 'Maksimalni povrat', 'Minimalni rizik']],
                                                        value='Sharpeov omjer'
                                                    ),
                                                    width=7
                                                )
                                            ],
                                            class_name='m-2'
                                        ),

                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Label("Odabir vremenskog raspona", className="form-label"),
                                                    width=5
                                                ),
                                                dbc.Col(
                                                    dcc.DatePickerRange(
                                                        id='date-range-picker',
                                                        min_date_allowed=pd.to_datetime('2018-01-01'),
                                                        max_date_allowed=pd.to_datetime('2023-12-31'),
                                                        start_date=pd.to_datetime('2021-01-01'),
                                                        end_date=pd.to_datetime('2023-07-01')
                                                    ),
                                                    width=7
                                                )
                                            ],
                                            class_name='m-2'
                                        ),

                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Label("Izvor podataka", className="form-label"),
                                                    width=5
                                                ),
                                                dbc.Col(
                                                    dcc.RadioItems(
                                                        options=[
                                                            {'label': 'Baza podataka', 'value': 'baza'},
                                                            {'label': 'ZG Burza (online)', 'value': 'zg_burza'},
                                                        ],
                                                        value='baza',
                                                        id='odabir_izvora_podataka'
                                                    ),
                                                    width=7
                                                )
                                            ],
                                            class_name='m-2'
                                        ),                 
                                    ]
                                ),
                            ]
                        ),
                        className='shadow border p-4',
                        style=custom_card_style
                    ),
                ], width=6),

                dbc.Col([
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Odaberite vrijednosnice:", className="form-label"),
                                                            ],
                                                            width=6
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Button('Dodaj sve', id='add-all', n_clicks=0, color="primary", className="col-auto"),
                                                            ],
                                                            width=6
                                                        ),
                                                    ],
                                                    class_name='mb-2'
                                                ),
                                                dcc.Dropdown(
                                                    id='stock-dropdown',
                                                    options=[{'label': stock, 'value': stock} for stock in get_stock_options()],
                                                    multi=True
                                                ),
                                            ],
                                            class_name='m-2'
                                        ),                 
                                    ]
                                ),
                            ]
                        ),
                        className='shadow border p-4',
                        style=custom_card_style
                    ),
                ], width=6),
            ],
            class_name='my-4'
        ),

        dbc.Row([
            dbc.Button('Optimiziraj portfelj', id='optimize-button', n_clicks=0, color="primary", className="mx-3"),
        ]),

        dcc.Loading(
            children=[
                dbc.Row([
                    dbc.Alert("", id='poruka', color="success", dismissable=True, is_open=False, className="mx-3"),
                ]),

                dbc.Row([
                    dbc.Col(html.Div(id='portfolio_graph', style={'display': 'flex', 'justify-content': 'center'}), width=10),
                ], align='center'),

                dbc.Row([
                    dbc.Col(dash_table.DataTable(id='portfolio_weights_table'), width=10),
                ], align='center'),

                dbc.Row([
                    dbc.Col(html.Div(id='efficient_frontier_graph', style={'display': 'flex', 'justify-content': 'center'}), width=10),
                ], align='center'),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='returns_graph'), width=10),
                ], align='center'),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='cum_returns_graph'), width=10),
                ], align='center'),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='data_graph'), width=10),
                ], align='center'),

                dbc.Row([
                    dbc.Col(dcc.Graph(id='correlation_graph'), width=10),
                ], align='center'),
                
                dbc.Row([
                    dbc.Col(dbc.Button('Skini izvještaj', id='excel_report', n_clicks=0, color="primary", className="col-auto m-4"), width=10),
                ], align='center'),

                dcc.Loading(
                    dbc.Row([
                        dcc.Download(id="excel-download"),
                    ]),
                )
            ],
            type="circle",
            fullscreen = True,
        )
    ]
)

Y = None
w = None

@app.callback(
    Output('excel-download', 'data'),
    Input('excel_report', 'n_clicks'),
    prevent_initial_call=True
)
def get_report(n_clicks):
    print('get_report POZVAN')

    global Y, w
    print(Y, w)

    if n_clicks > 0 and Y is not None and w is not None:
        report = rp.excel_report(Y, w) 

        time.sleep(5)
        file_path = os.path.join('', 'report.xlsx')
        
        return dcc.send_file(file_path, filename="report.xlsx")

    print('get_report ZAVRŠEN')
    return None

@app.callback(
    Output('date-range-picker', 'max_date_allowed'),
    Output('date-range-picker', 'end_date'),
    Input('dnevno_dohvacanje_podataka', 'n_intervals'),
    prevent_initial_call=True
)
def update_database(n):
    print('update_database POZVAN')
    try:
        print(f'Zadnji datum iz baze: {get_last_date_from_db()[:10]}')
        fill_database(start_time=get_last_date_from_db()[:10])
        print(f'Izvlačenje završeno, novi zadnji datum iz baze: {get_last_date_from_db()[:10]}')
    except Exception as e:
        print(traceback.print_exc())
        print('Došlo je do greške')

    zadnji_datum = pd.to_datetime(get_last_date_from_db()[:10])
    danasnji_datum = pd.to_datetime(datetime.now() + pd.Timedelta(days=1))

    print('update_database ZAVRŠEN')
    return danasnji_datum, zadnji_datum

@app.callback(
    Output('portfolio_graph', 'children'),
    Output('efficient_frontier_graph', 'children'),
    Output('portfolio_weights_table', 'data'),
    Output('returns_graph', 'figure'),
    Output('cum_returns_graph', 'figure'),
    Output('data_graph', 'figure'),
    Output('correlation_graph', 'figure'),
    Output('poruka', 'is_open'),
    Output('poruka', 'children'),
    Output('poruka', 'color'),
    Input('optimize-button', 'n_clicks'),
    State('odabir_izvora_podataka', 'value'),
    State('metric-dropdown', 'value'),
    State('date-range-picker', 'start_date'),
    State('date-range-picker', 'end_date'),
    State('stock-dropdown', 'value'),
    prevent_initial_call=True
)
def update_graph(n_clicks, izvor_podataka, metric, start_date, end_date, stocks):
    print('update_graph POZVAN')

    global Y, w

    bad_columns_message = []

    try:
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        
        if izvor_podataka == 'baza':
            prices, returns_data = read_from_db(stocks, start_date, end_date)
        else:
            prices, returns_data = get_stock_prices_and_returns(stocks, start_time=start_date, end_time=end_date)

        results_dict = transform_data(prices)

        nan_columns = results_dict['nan_columns']
        zero_return_columns = results_dict['zero_return_columns']

        Y = returns_data.copy()#results_dict['returns']
        Y.drop(nan_columns, axis=1, inplace=True)
        Y.drop(zero_return_columns, axis=1, inplace=True)

        if len(nan_columns) > 0:
            bad_columns_message.append(html.Br())
            bad_columns_message.append(f'Vrijednosnice koje nisu uzete u izračun zbog nedostatka podataka: {str(nan_columns)}')

        if len(zero_return_columns) > 0:
            bad_columns_message.append(html.Br())
            bad_columns_message.append(f'Vrijednosnice koje nisu uzete u izračun zbog konstatnog povrata u iznosu točno 0: {str(zero_return_columns)}')

        if Y.shape[1] < 2:
            print(Y)
            if Y.shape[1] == 1:
                raise Exception(f'Odabrana je samo jedna vrijednosnica koja ima valjane podatke: {Y.columns[0]}')
            raise Exception('Nije odabrana ni jedna vrijednosnica koja ima valjane podatke')
        
        port = rp.Portfolio(returns=Y)

        method_mu='hist'
        method_cov='hist'
        port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

        model='Classic'
        rm = 'MV' 
        obj = 'Sharpe'

        if metric == 'Minimalni rizik':
            obj = 'MinRisk'

        elif metric == 'Maksimalni povrat':
            obj = 'MaxRet'

        elif metric == '':
            obj = 'Utility'
            
        hist = True
        rf = 0
        l = 0
        
        w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

        weights_data = []
        for index, item in w['weights'].items():
            formatted_value = "{:.5f}".format(item*100)
            weights_data.append({'Vrijednosnica': index, 'Postotak uloga (%)': formatted_value})

        w = w[w.iloc[:, 0] > 0]

        fig = Figure(figsize=(5, 4))
        ax = fig.subplots()
        ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                    height=6, width=10, ax=ax)
        png_image = BytesIO()
        fig.savefig(png_image, format='png')
        encoded_image = base64.b64encode(png_image.getvalue()).decode('ascii')

        points = 20 
        frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
        label = 'Max Risk Adjusted Return Portfolio'
        mu = port.mu 
        cov = port.cov
        returns = port.returns 

        fig_frontier = Figure(figsize=(5, 4))
        ax_frontier = fig_frontier.subplots()
        ax_frontier = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                    rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                    marker='*', s=16, c='r', height=6, width=18, ax=ax_frontier)

        png_image_frontier = BytesIO()
        fig_frontier.savefig(png_image_frontier, format='png')
        encoded_image_frontier = base64.b64encode(png_image_frontier.getvalue()).decode('ascii')

        # Prikaz povrata
        fig_returns = go.Figure()
        for c in Y.columns:
            fig_returns.add_trace(go.Scatter(x=Y.index, y=Y[c]*100, mode='markers+lines', name=c))
            
        fig_returns.update_layout(
            title="Povrati odabranih vrijednosnica kroz vrijeme",
            xaxis_title="Vrijeme",
            yaxis_title="Povrati (%)",
        )

        cumulative_returns = (1 + Y).cumprod() - 1
        portfolio_cumulative_returns = (1 + Y.mul(w.values.T).sum(axis=1)).cumprod() - 1

        # Prikaz kumulatinih povrata
        fig_cum_returns = go.Figure()
        fig_cum_returns.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns*100, mode='markers+lines', name='Portfelj'))

        for c in cumulative_returns.columns:
            fig_cum_returns.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[c]*100, mode='markers+lines', name=c))
            
        fig_cum_returns.update_layout(
            title="Kumulativni povrati odabranih vrijednosnica kroz vrijeme",
            xaxis_title="Vrijeme",
            yaxis_title="Kumulativni povrati (%)",
        )

        # Prikaz cijena
        fig_price = go.Figure()
        for c in prices.columns:
            fig_price.add_trace(go.Scatter(x=prices.index, y=prices[c], mode='markers+lines', name=c))
            
        fig_price.update_layout(
            title="Cijene odabranih vrijednosnica kroz vrijeme u eurima",
            xaxis_title="Vrijeme",
            yaxis_title="€",
        )

        fig_correlation = px.imshow(Y.corr(), title='Korelacijska matrica povrata')

        fig_correlation.update_layout(
            autosize=False,
            width=800,
            height=800,
        )

        print('update_graph ZAVRŠEN')
        return (html.Img(src='data:image/png;base64,{}'.format(encoded_image), height=600, width=800),  # smaller size
                html.Img(src='data:image/png;base64,{}'.format(encoded_image_frontier), height=600, width=800),  # smaller size
                weights_data,
                fig_returns,
                fig_cum_returns,
                fig_price,
                fig_correlation,
                True,
                ["Porftelj je uspješno optimiziran!"] + bad_columns_message,
                "success"
                )

    except Exception as e:
        print(traceback.print_exc())
        message = str(e)
        if not stocks:
            message = 'Nema odabranih vrijednosnica za odabir'
        # raise PreventUpdate
        print('update_graph ZAVRŠEN')
        return (None,
            None,
            None,
            None,
            None,
            None,
            go.Figure(),
            True,
            [message] + bad_columns_message,
            'danger'
        )
    
@app.callback(
    Output('stock-dropdown', 'value'),
    Input('add-all', 'n_clicks'),
    State('stock-dropdown', 'options'),
    prevent_initial_call=True
)
def add_all_stocks(n, options):
    # if button is clicked, return all stocks
    if n > 0:
        return [option['value'] for option in options]
    else:
        raise PreventUpdate

if __name__ == '__main__':
    print(f'Zadnji datum iz baze: {get_last_date_from_db()[:10]}')
    fill_database(start_time=get_last_date_from_db()[:10])
    print(f'Izvlačenje završeno, novi zadnji datum iz baze: {get_last_date_from_db()[:10]}')

    time.sleep(3)
    print('POKRETANJE SERVERA')
    app.run_server(debug=False, port=8052)
