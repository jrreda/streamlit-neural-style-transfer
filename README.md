# PyTorch Fast Neural Style Transfer 🎨 

This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting.

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). The saved-models for examples shown in the README can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0).

## Running App
1. First install all the dependencies
    ```
    pip install -r requirements.txt
    ```

2. Run Streamlit app

    Demo your model by running [app.py](app.py)
    ```
    streamlit run app.py
    ```

![demo](https://github.com/jrreda/streamlit-style-transfer/blob/main/screenplay.webm)
<video src='https://github.com/jrreda/streamlit-style-transfer/blob/main/screenplay.webm' width="500" height="500" autoplay muted>
