import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import dash_table

# Load the data

lda_5_0=[('life', 0.009688808), ('film', 0.009116198), ('movie', 0.007929486), ('one', 0.006486829), ('story', 0.0060272547), ('love', 0.0056743673), ('woman', 0.005012887), ('time', 0.004861408), ('year', 0.004464996), ('people', 0.0043770815), ('like', 0.0040740506), ('family', 0.00393733), ('see', 0.0038841749), ('young', 0.003823651), ('man', 0.0036989632)]
lda_5_1=[('film', 0.011182349), ('one', 0.008754396), ('movie', 0.008601488), ('like', 0.006708311), ('action', 0.006374844), ('get', 0.005459211), ('scene', 0.0054525295), ('time', 0.004308448), ('good', 0.0038025498), ('horror', 0.0036449935), ('plot', 0.00355663), ('really', 0.003524779), ('well', 0.0035001931), ('story', 0.0034288906), ('go', 0.0033313476)]
lda_5_2=[('film', 0.0069987364), ('one', 0.0066415155), ('movie', 0.00586016), ('life', 0.0043752827), ('love', 0.0043438743), ('get', 0.004117156), 
('action', 0.003802284), ('like', 0.0035588741), ('girl', 0.0035514005), ('school', 0.0035014942), ('character', 0.003500304), ('also', 0.0034539588), ('time', 0.003388058), ('man', 0.0032924428), ('father', 0.003071858)]
lda_5_3=[('one', 0.0058496306), ('film', 0.005509468), ('man', 0.005405354), ('life', 0.005122721), ('movie', 0.0036873615), ('also', 0.003375457), ('time', 0.0032708459), ('story', 0.0031543397), ('world', 0.003053581), ('find', 0.0029347087), ('take', 0.0029196134), ('family', 0.0028779306), ('young', 0.0028768003), ('year', 0.0028288178), ('new', 0.0026959819)]
lda_5_4=[('film', 0.030860735), ('story', 0.010828121), ('character', 0.010248775), ('one', 0.0078984285), ('life', 0.006404216), ('well', 0.0062720464), ('movie', 0.0057682707), ('time', 0.0046749027), ('love', 0.0046551656), ('performance', 0.0043680253), ('also', 0.004217273), ('make', 
0.0039468217), ('great', 0.0036811493), ('good', 0.003660835), ('scene', 0.0035626094)]


lda_10_0=[('film', 0.012318197), ('like', 0.008056354), ('get', 0.0061276536), ('one', 0.0056405803), ('even', 0.005013233), ('time', 0.0049831476), ('year', 0.0039690128), ('sex', 0.0037546484), ('make', 0.0036858679), ('see', 0.0034732483), ('old', 0.0033510118), ('take', 0.003330177), ('would', 0.0031794638), ('girl', 0.0031454763), ('little', 0.0031235616)]
lda_10_1=[('film', 0.034319706), ('character', 0.017680397), ('story', 0.009709875), ('one', 0.007802077), ('well', 0.0075585297), ('really', 0.0067016343), ('feel', 0.006005766), ('time', 0.0057561905), ('much', 0.00574007), ('make', 0.0053035375)]
lda_10_2=[('film', 0.040267777), ('character', 0.0134050045), ('story', 0.011373803), ('one', 0.009374851), ('well', 0.008536189), ('movie', 0.008302986), ('time', 0.0058063245), ('good', 0.0056024604), ('also', 0.0052070133), ('scene', 0.0049760495)]
lda_10_3=[('one', 0.007075407), ('father', 0.0069550998), ('mother', 0.0064421245), ('get', 0.006419316), ('wife', 0.0060213935), ('life', 0.0056743603), ('daughter', 0.0056198216), ('girl', 0.0056009297), ('family', 0.005275215), ('movie', 0.0052186465)]
lda_10_4=[('life', 0.008949764), ('get', 0.007100661), ('one', 0.006956305), ('woman', 0.0067123757), ('girl', 0.0058163037), ('man', 0.0054900944), ('year', 0.005158351), ('go', 0.0050076405), ('friend', 0.004874394), ('like', 0.00479343)]

lda_20_0=[('film', 0.022224033), ('movie', 0.017749226), ('like', 0.01738852), ('bad', 0.013285457), ('one', 0.010169052), ('really', 0.009461171), ('even', 0.009336733), ('good', 0.007817627), ('would', 0.0071287197), ('character', 0.007107743)]
lda_20_1=[('film', 0.008530606), ('life', 0.008394166), ('story', 0.0065801307), ('time', 0.006571339), ('take', 0.0061238166), ('one', 0.0060834307), ('find', 0.0060711354), ('get', 0.005632654), ('world', 0.005534351), ('u', 0.005043636)]
lda_20_2=[('song', 0.021261407), ('film', 0.01816529), ('music', 0.018150676), ('movie', 0.014381695), ('great', 0.009318496), ('one', 0.008254762), ('role', 0.007921041), ('star', 0.007800076), ('performance', 0.007486062), ('best', 0.007263295)]
lda_20_3=[('killer', 0.0127915945), ('kill', 0.009221011), ('film', 0.008907955), ('one', 0.007539822), ('get', 0.0070353122), ('chan', 0.0065807113), ('man', 0.0065544304), ('brother', 0.0064739953), ('killed', 0.0059537464), ('take', 0.005134298)]
lda_20_4=[('movie', 0.1038846), ('like', 0.018182518), ('one', 0.01407306), ('watch', 0.014037391), ('really', 0.013592972), ('good', 0.0107544735), ('time', 0.010521509), ('see', 0.00950197), ('would', 0.008739316), ('first', 0.007700532)]


# Define the colors
colors = {
    'background': '#F4F6F9',
    'text': '#5F5F5F'
}

custom_font = "'PT Sans', sans-serif"

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children='Machine Learning Application - Final Project',
                    style={
                        'textAlign': 'center',
                        'color': colors['text']
                    }
                ),
                html.P(
                    children='Analysis of film reviews',
                    style={
                        'textAlign': 'center',
                        'color': colors['text'],
                        'fontSize': 24,
                        'fontFamily': custom_font
                    }
                )
            ],
            style={
                'backgroundColor': colors['background'],
                'padding': '2rem'
            }
        ),
        html.Div(
        children=[
        html.H1(
            children='LDA Topic Modeling on the film reviews',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
        html.Div(
            children=[
                dcc.Dropdown(
                    id='variable-dropdown',
                    options=[
                        {'label': 'number of topics = 5 ', 'value': 5},
                        {'label': 'number of topics = 10', 'value': 10},
                        {'label': 'number of topics = 20', 'value': 20}
                    ],
                    style={'width': '200px'},
                    value=5
                )
            ],
            style={
                'backgroundColor': colors['background'],
                'padding': '2rem',
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
            }
        ),
        html.Div(
            id='table-container',
            style={
                'backgroundColor': colors['background'],
                'padding': '2rem'
            }
        )
    ]
)

        
    ]
)

# Define the callback
@app.callback(
    dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('variable-dropdown', 'value')]
)
def update_table(value):
    if value == 5:
        list_data1=[[item[0], item[1]] for item in lda_5_0]
        df1=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data2=[[item[0], item[1]] for item in lda_5_1]
        df2=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data3=[[item[0], item[1]] for item in lda_5_2]
        df3=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data4=[[item[0], item[1]] for item in lda_5_3]
        df4=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data5=[[item[0], item[1]] for item in lda_5_4]
        df5=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
    elif value == 10:
        list_data1=[[item[0], item[1]] for item in lda_10_0]
        df1=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data2=[[item[0], item[1]] for item in lda_10_1]
        df2=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data3=[[item[0], item[1]] for item in lda_10_2]
        df3=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data4=[[item[0], item[1]] for item in lda_10_3]
        df4=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data5=[[item[0], item[1]] for item in lda_10_4]
        df5=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
    else:
        list_data1=[[item[0], item[1]] for item in lda_20_0]
        df1=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data2=[[item[0], item[1]] for item in lda_20_1]
        df2=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data3=[[item[0], item[1]] for item in lda_20_2]
        df3=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data4=[[item[0], item[1]] for item in lda_20_3]
        df4=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
        list_data5=[[item[0], item[1]] for item in lda_20_4]
        df5=pd.DataFrame(list_data1, columns=['Token', 'Weight'])
    table1= dash_table.DataTable( id='weight-table1', columns=[{"name": i, "id": i} for i in df1.columns], data =df1.to_dict('records'), style_header={ 'border': '1px solid black' })
    table2= dash_table.DataTable( id='weight-table1', columns=[{"name": i, "id": i} for i in df1.columns], data =df2.to_dict('records'), style_header={ 'border': '1px solid black'})
    table3= dash_table.DataTable( id='weight-table1', columns=[{"name": i, "id": i} for i in df1.columns], data =df3.to_dict('records'), style_header={ 'border': '1px solid black'})
    table4= dash_table.DataTable( id='weight-table1', columns=[{"name": i, "id": i} for i in df1.columns], data =df4.to_dict('records'), style_header={ 'border': '1px solid black'})
    table5= dash_table.DataTable( id='weight-table1', columns=[{"name": i, "id": i} for i in df1.columns], data =df5.to_dict('records'), style_header={ 'border': '1px solid black'})
    tables=[table1,table2,table3,table4,table5]
    return tables

if __name__ == '__main__':
    app.run_server(debug=True)