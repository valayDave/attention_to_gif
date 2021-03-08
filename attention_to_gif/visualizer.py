# importing matplot lib 
import matplotlib.pyplot as plt 
import numpy as np 
import torch
# importig movie py libraries 
from moviepy.editor import VideoClip 
from moviepy.video.io.bindings import mplfig_to_npimage 

class AttentionVisualizer:
    """
    Creates a GIF of the transition of attention weights across the layers. 
    layer_wise_attention_weights : Tensor : (num_layers,batch_size,num_heads,seq_len_x,seq_len_y)
    x_label_toks : labels for indices in the sequence.
    y_label_toks : labels for indices in the sequence.
    title_message: Message to print in the video/Gif
    chosen_head : int : a particular to head to visualise attention for. 
              If None: Attention is a summed for all heads at a layer
    seq_len_x_lim : constrain the size of of head in the x dimension 
    seq_len_y_lim : constrain the size of of head in the y dimension 
    """
    def __init__(self,\
                layer_wise_attention_weights:torch.Tensor,
                seq_len_x_lim=None,
                seq_len_y_lim=None,
                chosen_item=0,
                chosen_head=None,
                x_label_toks=[],
                y_label_toks=[],
                fig_size=(10,10),
                title_message='',
                ) -> None:
        #()
        self.num_layers, self.batch_size , self.num_attention_heads , seq_len_x,seq_len_y = layer_wise_attention_weights.size()
        # Doing this ensure that it work.
        self.seq_len_x_lim = seq_len_x_lim
        self.seq_len_y_lim  = seq_len_y_lim
        self.chosen_item=chosen_item
        self.layer_wise_attention_weights = layer_wise_attention_weights
        self.fig, self.ax = plt.subplots(figsize=fig_size)
        self.x_label_toks = x_label_toks
        self.y_label_toks = y_label_toks
        self.title_message = title_message
        self.chosen_head = chosen_head
        # self.fig.colorbar()
    
    def get_attention_values(self,layer,chosen_head=None):
      if chosen_head is not None:
        conv_arr = self.layer_wise_attention_weights[int(layer)][self.chosen_item][chosen_head].cpu().numpy()
      else:
        conv_arr = self.layer_wise_attention_weights[int(layer)][self.chosen_item].sum(dim=0).cpu().numpy()
      if self.seq_len_x_lim is not None:
          conv_arr= conv_arr[:,:self.seq_len_x_lim]
      if self.seq_len_y_lim is not None:
          conv_arr= conv_arr[:self.seq_len_y_lim]
      
      return conv_arr

    
    def __call__(self,t):
        # clear 
        if len(self.ax.images) > 0:
          self.ax.images[-1].colorbar.remove()
        self.ax.clear()

        conv_arr = self.get_attention_values(t,chosen_head=self.chosen_head)
        cax = self.ax.matshow(conv_arr,origin='lower', cmap='viridis',aspect='auto')
        # self.y_label_toks
        self.fig.colorbar(cax,ax=self.ax)
        
        if len(self.x_label_toks) > 0:
            self.ax.set_xticks([i for i in range(len(self.x_label_toks))])
            self.ax.set_xticklabels(self.x_label_toks,)

        if len(self.y_label_toks) > 0:
            self.ax.set_yticks([len(self.y_label_toks)-i-1 for i in range(len(self.y_label_toks))])
            self.ax.set_yticklabels(self.y_label_toks)

        default_title = f" Attention At Layer : {int(t)} \n"
        if self.chosen_head is not None:
          default_title = f" Attention At Layer : {int(t)} And Head : {self.chosen_head}\n"

        if self.title_message is '':
          self.ax.set_title(default_title)
        else:
          self.ax.set_title(f"{self.title_message}\n {default_title}")
          
        return mplfig_to_npimage(self.fig)

    def save_visualisation(self,viz_name='attention_viz.gif',fps=20):
        animation = VideoClip(make_frame=self,duration=self.num_layers) 
        animation.write_gif(viz_name,fps=fps)

    def show_visualisation(self,viz_name='attention_viz.gif',fps = 20, loop = False, autoplay = False):
        # animation = VideoClip(self, duration = self.num_layers) 
        animation = VideoClip(make_frame=self,duration=self.num_layers) 
        animation.ipython_display(fps =fps ,loop=loop,autoplay=autoplay) 

    def create_single_plot(self, fig_size=(10,10)):
        fig,axes = plt.subplots(nrows=self.num_layers, ncols=self.num_attention_heads,figsize=fig_size)
        for lidx,layerax in enumerate(axes):
          for hidx,headax in enumerate(layerax):
            conv_arr = self.get_attention_values(lidx,chosen_head=hidx)
            
            cax = headax.matshow(conv_arr,origin='lower', cmap='viridis',aspect='auto')
            fig.colorbar(cax,ax=headax)
            if len(self.x_label_toks) > 0:
              headax.set_xticks([i for i in range(len(self.x_label_toks))])
              headax.set_xticklabels(self.x_label_toks,)

            if len(self.y_label_toks) > 0:
              headax.set_yticks([len(self.y_label_toks)-i-1 for i in range(len(self.y_label_toks))])
              headax.set_yticklabels(self.y_label_toks)

            default_title = f" Attention At Layer : {int(lidx)} And Head : {hidx}\n"
            headax.set_title(default_title)
        return fig