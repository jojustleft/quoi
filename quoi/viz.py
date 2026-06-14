import plotly.graph_objects as go

config_1 = dict(text='#201B23',
                 background='#FCF6F5',
                 grid='#ebe1df',
                 entries=['#5C825B', '#D5834F', '#4E76B2', '#904A58', '#C0E0DE'],
                 serif_font='Georgia',
                 default_font='Helvetica')

layout = dict(
    layout = go.Layout(
        paper_bgcolor=config_1['background'],
        plot_bgcolor=config_1['background'],
        title=dict(font=dict(family=config_1['serif_font'], size=22, color=config_1['text']),
                   xanchor='left', xref='paper', x=0,
                   yanchor='bottom', yref='container', y=0.94,
        ),
        xaxis=dict(title=dict(font=dict(family=config_1['default_font'], size=16, color=config_1['text'])),
                   showline=True, ticks='outside', mirror=True, zeroline=False,
                   gridcolor=config_1['grid']),
        yaxis=dict(title=dict(font=dict(family=config_1['default_font'], size=16, color=config_1['text'])),
                   showline=True, ticks='outside', mirror=True, zeroline=False,
                   gridcolor=config_1['grid']),
        legend=dict(title=dict(font=dict(family=config_1['default_font'], size=15, color=config_1['text'])),
                    orientation='h',
                    xanchor='right',
                    x=1,
                    yanchor='bottom',
                    y=1.02,
                    font=dict(family=config_1['default_font'])
        ),
        colorway=config_1['entries'],
        margin=dict(t=75, b=75, l=75, r=75, pad=5)
    )
)


def build_subtitle(s):
    # Not the most elegant solution, but it is the approach which provided the layout I preferred
    return f"<br><span style='font-family: Helvetica; font-size: 20px; color:#4a464d'><sup>{s}</sup></span>"