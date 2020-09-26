""" file containing utility functions e.g. plotting 
"""
import pandas as pd 
import numpy as np 
import plotly.graph_objs as go
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