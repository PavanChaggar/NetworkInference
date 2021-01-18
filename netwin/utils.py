""" file containing utility functions e.g. plotting 
"""
import pandas as pd 
import numpy as np 
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def get_nodes(scale):
   network_map = {
      83: '/home/chaggar/Documents/Network_Inference/data/mni_coordinates/mni-parcellation-scale1_coordinates.csv'
   }
   return network_map[scale]

def max_norm(in_vector): 
   return in_vector / np.max(in_vector)

def sum_mean(in_vector):
   return in_vector / np.sum(in_vector)

def plot_nodes(in_vector, opacity, colour='Reds'):

   nodes = pd.read_csv(get_nodes(len(in_vector)))

   x, y, z = np.array(nodes.x), np.array(nodes.y), np.array(nodes.z)

   trace1=go.Scatter3d(x=x,
                  y=y,
                  z=z,
                  mode='markers', 
                  name='actors',
                  marker=dict(symbol='circle',
                              size=20,
                              color=max_norm(in_vector),
                              colorscale=colour,
                              line=dict(color='rgba(50,50,50,0.3)', width=0.5),
                              opacity=opacity,
                              colorbar=dict(
                                    thickness=15,
                                    title='Node Connections',
                                    xanchor='left',
                                    titleside='right'),
                              ),                              
                  text=nodes.Label,
                  hoverinfo='text'
                  )

   axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

   layout = go.Layout(
            title="Connectome",
            width=1000
            ,
            height=500,
            showlegend=False,
            scene=dict(
               xaxis=dict(axis),
               yaxis=dict(axis),
               zaxis=dict(axis),
         ),
      margin=dict(
         t=100
      ),
      hovermode='closest',
      )

   data=[trace1]
   fig=go.Figure(data=data, layout=layout)
   return fig.show()

def plot_timeseries(data, time_interval, colours=['g'], alpha=[0.5]):

    ax = plt.figure()

    for i, x in enumerate(data):
        for j in range(len(x[0])):
            plt.plot(time_interval, x[:,j], c=colours[i], alpha=alpha[i])

    return ax.show()

def contour_2dmvn(means, cov, n):
    samples = np.random.multivariate_normal(means, cov, n)
    (counts, x_bins, y_bins) = np.histogram2d(samples[:, 0], samples[:, 1])

    fig = go.Figure(data =
        go.Contour(
            z=counts,
            x=x_bins, # horizontal axis
            y=y_bins# vertical axis
        ))
    return fig.show()

def barplot_concentrations(true, inferred, opacity=0.7):
    fig = go.Figure(data=[
        go.Bar(name='True', x=np.arange(len(true)), y=true, opacity=opacity),
        go.Bar(name='Inferred', x=np.arange(len(inferred)), y=inferred, opacity=opacity)
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    return fig.show()

def plot_2dmvn(means, cov, n=10000):
    samples = np.random.multivariate_normal(means, cov, n)
    (counts, x_bins, y_bins) = np.histogram2d(samples[:, 0], samples[:, 1])

    fig = make_subplots(rows=2, cols=2, column_widths=[0.8,0.2], row_heights=[0.2,0.8])

    fig.add_trace(
        go.Contour(
            z=counts,
            x=x_bins, # horizontal axis
            y=y_bins,# vertical axis
            showlegend=False,
            showscale=False,
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Histogram(x=samples[:,0],showlegend=False),
        row=1, col=1
    )


    fig.add_trace(
        go.Histogram(y=samples[:,1],showlegend=False),
        row=2, col=2
    )

    fig.update_layout(height=600, width=600)
    return fig.show()