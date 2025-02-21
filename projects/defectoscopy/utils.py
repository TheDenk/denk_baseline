import numpy as np


def construct(signal, signal_h=1284):
    channel_dict = {0:0, 1:4, 2:1, 3:5, 4:2, 5:6, 6:3, 7:7}
    channel_clip_map = {0:185, 1:136, 2:136, 3:185, 4:185, 5:136, 6:136, 7:185}
    channel_bias_map = {0:0, 1:506, 2:370, 3:185, 4:642, 5:1148, 6:1012, 7:827}

    coord_synth_map = signal['coord_display']\
                     .drop_duplicates()\
                     .reset_index(drop=True).reset_index()\
                     .rename(columns={'index':'coord_synth'})

    signal = signal.merge(coord_synth_map, on='coord_display', how='left')

    signal = signal[['coord_synth', 'channel', 'amplitude', 'delay']].astype({'coord_synth': int,
                                                                              'channel': np.uint8,
                                                                              'amplitude': np.uint8,
                                                                              'delay': np.uint8})
    # Measurement correction
    # channel
    signal['channel'] = signal['channel'].map(channel_dict)
    # amplitude
    signal['amplitude']  = signal['amplitude'].clip(1, 15)
    # delay
    signal['delay'] = signal['delay'].clip(0, signal['channel'].map(channel_clip_map))
    signal['delay'] = signal['delay'] - 1 + signal['channel'].map(channel_bias_map)

    # Transformation measurements into matrix coordinate x delay   
    signal_rep = np.zeros(shape=(signal['coord_synth'].nunique(), signal_h), dtype=np.uint8) 
    signal_rep[signal['coord_synth'].values, signal['delay'].values] = signal['amplitude'].values
    return signal_rep