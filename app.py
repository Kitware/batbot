import time

import cv2
import gradio as gr
import numpy as np

import batbot
from batbot import classifier

USGS = [
    'USGS',
    int(classifier.CONFIGS['usgs']['thresh'] * 100),
]


def predict(filepath, config, classifier_thresh):
    start = time.time()

    if config == 'USGS':
        config = 'usgs'
    else:
        raise ValueError()

    classifier_thresh /= 100.0

    # Load data
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    pixels = h * w
    megapixels = pixels / 1e6

    classifier_, detects = batbot.pipeline(
        filepath,
        config=config,
        classifier_thresh=classifier_thresh,
    )

    output = []
    for detect in detects:
        label = detect['l']
        conf = detect['c']
        point1 = (
            int(np.around(detect['x'])),
            int(np.around(detect['y'])),
        )
        point2 = (
            int(np.around(detect['x'] + detect['w'])),
            int(np.around(detect['y'] + detect['h'])),
        )
        color = (255, 0, 0)
        img = cv2.rectangle(img, point1, point2, color, 2)
        output.append(f'{label}: {conf:0.04f}')
    output = '\n'.join(output)

    end = time.time()
    duration = end - start
    speed = duration / megapixels
    speed = f'{speed:0.02f} seconds per megapixel (total: {megapixels:0.02f} megapixels, {duration:0.02f} seconds)'

    return img, speed, classifier_, output


interface = gr.Interface(
    fn=predict,
    title='BatBot',
    inputs=[
        gr.Image(type='filepath'),
        gr.Radio(
            label='Model Configuration',
            type='value',
            choices=[USGS[0]],
            value=USGS[0],
        ),
        gr.Slider(label='Classifier Confidence Threshold', value=USGS[1]),
    ],
    outputs=[
        gr.Image(type='numpy'),
        gr.Textbox(label='Prediction Speed', interactive=False),
        gr.Number(label='Predicted Classifier Confidence', precision=5, interactive=False),
        gr.Textbox(label='Predicted Detections', interactive=False),
    ],
    examples=[
        ['examples/example.wav'] + USGS,
    ],
    cache_examples=True,
    allow_flagging='never',
)

interface.launch(server_name='0.0.0.0')
