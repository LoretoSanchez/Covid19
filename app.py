import pandas as pd
import numpy as np
#import cufflinks as cf
import plotly
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime,timedelta

from scipy.optimize import curve_fit
import sklearn.metrics as sklm
from scipy.optimize import fsolve

import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls

username = 'LoreChile'
api_key = 'NIIlk6BYFSCK5383WlEe'

chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

#########################################################################################
# Función de gráfico 
def graficar(fig, titulo,xtitulo,ytitulo,tipo,dff,ejex):
    
    if ejex == 'tiempo':
        updatemenus = list([
            dict(active=1,
                 buttons=list([
                    dict(label='Log',
                         method='update',
                         args=[{'visible': [True, True]},
                               {'title': titulo+', escala logarítmica',
                                'yaxis': {'type': 'log','title':ytitulo},}]),
                    dict(label='Lineal',
                         method='update',
                         args=[{'visible': [True, True]},
                               {'title': titulo+', escala lineal',
                                'yaxis': {'type': 'linear','title':ytitulo},}])]),
                    direction="down",pad={"r": 0, "t": 0},showactive=True,
                    x=-0.16,xanchor="left",y=1.05,yanchor="top")])
   
    elif ejex == 'num':
        updatemenus = list([
            dict(active=1,
                 buttons=list([
                    dict(label='Log',
                         method='update',
                         args=[{'visible': [True, True]},
                               {'title': titulo+', escala logarítmica',
                                'yaxis': {'type': 'log','title':ytitulo},
                                'xaxis': {'type': 'log','title':xtitulo}}]),
                    dict(label='Lineal',
                         method='update',
                         args=[{'visible': [True, True]},
                               {'title': titulo+', escala lineal',
                                'yaxis': {'type': 'linear','title':ytitulo},
                                'xaxis': {'type': 'linear','title':xtitulo}}])]),
                    direction="down",pad={"r": 0, "t": 0},showactive=True,
                    x=-0.16,xanchor="left",y=1.05,yanchor="top")])  
    
    elif ejex == 'lin_log':
        updatemenus = list([
            dict(active=1,
                 buttons=list([
                    dict(label='Log',
                         method='update',
                         args=[{'visible': [True, True]},
                               {'title': titulo+' (escala logarítmica)',
                                'yaxis': {'type': 'linear','title':ytitulo},
                                'xaxis': {'type': 'log','title':xtitulo}}]),
                    dict(label='Lineal',
                         method='update',
                         args=[{'visible': [True, True]},
                               {'title': titulo+' (escala lineal)',
                                'yaxis': {'type': 'linear','title':ytitulo},
                                'xaxis': {'type': 'linear','title':xtitulo}}])]),
                    direction="down",
                    pad={"r": 0, "t": 0},
                    showactive=True,
                    x=-0.16,
                    xanchor="left",
                    y=1.05,
            yanchor="top")])        
        
        
    fig.layout = dict(updatemenus=updatemenus,title=titulo+', escala lineal')
    
    if tipo=='dframe':
        layout = dict(updatemenus=updatemenus,title=titulo+', escala lineal')
        fig = dff.iplot(asFigure=True,layout=layout)
    
    fig.update_layout(xaxis_title=xtitulo,yaxis_title=ytitulo,
                      font=dict(family="Courier New, monospace",size=12,color="#7f7f7f"))
    
    return fig
    #fig.show()
#########################################################################################

paises = ['Chile','China','USA','Spain','Italy','Germany','UK','South Korea','Brazil','Argentina','Peru',
        'Ecuador','New Zealand','Australia']

#########################################################################################

# paletas de coloresplotly discretos:
category10 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
dark2 = ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666"]
tableau10 = ["#4e79a7","#f28e2c","#e15759","#76b7b2","#59a14f","#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab"]

# Generador de colores para graficar
from typing import Iterable, Tuple
import colorsys
import itertools
from fractions import Fraction
from pprint import pprint

def zenos_dichotomy() -> Iterable[Fraction]:
    for k in itertools.count():
        yield Fraction(1,2**k)

def fracs() -> Iterable[Fraction]:

    yield Fraction(0)
    for k in zenos_dichotomy():
        i = k.denominator # [1,2,4,8,16,...]
        for j in range(1,i,2):
            yield Fraction(j,i)

# can be used for the v in hsv to map linear values 0..1 to something that looks equidistant
# bias = lambda x: (math.sqrt(x/3)/Fraction(2,3)+Fraction(1,3))/Fraction(6,5)
HSVTuple = Tuple[Fraction, Fraction, Fraction]
RGBTuple = Tuple[float, float, float]

def hue_to_tones(h: Fraction) -> Iterable[HSVTuple]:
    for s in [Fraction(6,10)]: # optionally use range
        for v in [Fraction(8,10),Fraction(5,10)]: # could use range too
            yield (h, s, v) # use bias for v here if you use range

def hsv_to_rgb(x: HSVTuple) -> RGBTuple:
    return colorsys.hsv_to_rgb(*map(float, x))

flatten = itertools.chain.from_iterable

def hsvs() -> Iterable[HSVTuple]:
    return flatten(map(hue_to_tones, fracs()))

def rgbs() -> Iterable[RGBTuple]:
    return map(hsv_to_rgb, hsvs())

def rgb_to_css(x: RGBTuple) -> str:
    uint8tuple = map(lambda y: int(y*255), x)
    return "rgb({},{},{})".format(*uint8tuple)

def css_colors() -> Iterable[str]:
    return map(rgb_to_css, rgbs())

def ncolores (n):
    pycolores = category10+dark2+tableau10
    num=0
    if n<=28:
        mun =1
    else: 
        num= n-28
    clrs = list(itertools.islice(css_colors(), num))
    pycolores.extend(clrs)
    return pycolores

#########################################################################################
# Datos Covid-19 Internacionales

# Función que lee archivos csv con datos de Chile
df_lee = pd.read_csv('chile.csv', sep=',')
df_regiones = df_lee.copy()
df_chile = df_lee.loc[df_lee['Region'] == 'Chile'].reset_index(drop = True)
df_chile.rename(columns={"Region": "Pais"}, inplace =True)
df_chile['Fecha']= pd.to_datetime(df_chile['Fecha'])
#df_chile.tail(2)

# Función que lee archivos csv con datos internacionales y genera dataframes por país
df_lee = pd.read_csv('paises.csv', sep=',')

# datos desde JHU (Johns Hopkins University)
df_paises = dict()
df_paises['Chile'] = df_chile# datos de confirmados de minsal
for p in paises[1:]:
    df_tmp = df_lee.loc[df_lee['Pais'] == p].reset_index(drop=True)
    df_tmp['Fecha']= pd.to_datetime(df_tmp['Fecha'])
    df_paises[p]=df_tmp

# datos desde OWID (Our World in Data)
df_lee2 = pd.read_csv('paises2.csv', sep=',')
paises2 = df_lee2['Pais'].unique()

df_paises2 = dict()
for p in paises2:
    df_tmp = df_lee2.loc[df_lee2['Pais'] == p].reset_index(drop=True)
    df_tmp['Fecha']= pd.to_datetime(df_tmp['Fecha'])
    df_paises2[p]=df_tmp

df_paises['Spain']['Contagios']=df_paises2['Spain']['Contagios']
#########################################################################################

# Generación de un diccionario con los dataframes de casos confirmados de los distintos  
# países con día 0 como el día en que superaron los 100 contagiados.
dic_df_conf_int = dict()
for i,p in enumerate(paises):
    #print("pais: ",p)
    
    df_tmp = df_paises[p][['Fecha','Contagios']].copy()
    #print(df_tmp)
    idx = np.nonzero(np.array(list(df_tmp['Contagios'].values)) > 100)#índice del 1er día con + de 100 contagios
    df_tmp = df_tmp[idx[0][0]:]
    df_tmp = df_tmp.reset_index(drop=True).reset_index()
    df_tmp.columns = ['Dia', 'Fecha','Contagios']  
    #print(p,': ',df_tmp.head())
    dic_df_conf_int[p] = df_tmp

# Gráfico de contagios confirmados acumulados en los distintos países por fecha desde 1er contagio
colores = ncolores(len(paises))   #list(itertools.islice(css_colors(), len(pais)))
fig = go.Figure()
for i,p in enumerate(paises):
    if i==0:
        ancho =3
    else:
        ancho =2
    fig.add_trace(go.Scatter(x=df_paises[p]['Fecha'],y=df_paises[p]['Contagios'],
            line=dict(width=ancho, color=colores[i]),name=p,
            hovertemplate = 'Casos: %{y:.0f}'+'<br>Fecha: %{x}<br>')) #País: '+ pais[p]+<br
titulo = 'Número de Casos Confirmados Internacionales'
xtitulo = "Fecha"
ytitulo = 'Casos Confirmados'
fig_contagios_mundo = graficar(fig, titulo,xtitulo,ytitulo,'itera',0,'tiempo')#px.colors.qualitative.Alphabet[p]
#cases_int_time = fig
#fig.show()
#py.plot(cases_int_time, filename = 'Casos en el tiempo', auto_open=False)


########### Initiate the app
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#server = app.server
#app.title=tabtitle


########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading),
    dcc.Graph(
        id='contagios_mundo',
        figure=fig_contagios_mundo
    ),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A('Data Source', href=sourceurl),
    ]
)







if __name__ == '__main__':
    app.run_server()